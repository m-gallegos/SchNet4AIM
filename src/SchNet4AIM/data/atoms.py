"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os
import warnings
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import SchNet4AIM as s4aim
from SchNet4AIM import Properties
from SchNet4AIM.environment import SimpleEnvironmentProvider, collect_atom_triples
from tqdm import tqdm
import ase as ase
import json as json

logger = logging.getLogger(__name__)

__all__ = [
    "AtomsData",
    "AtomsDataSubset",
    "AtomsDataError",
    "AtomsConverter",
    "get_center_of_mass",
    "get_center_of_geometry",
    "json_2_ase_atoms",
]

def json_2_ase_atoms(jsondb,idx):
    """
    Creates a standard ase atoms object starting
    from a json database file

    Args:
        jsondb: json database file
        idx   : id of the instance in the database
    Returns:
        ASE atoms object
    """
    label=''.join(jsondb["ele"][idx])
    cords = jsondb["pos"][idx]
    ase_atoms=ase.Atoms(label,cords)
    return ase_atoms

def get_center_of_mass(atoms):
    """
    Computes center of mass.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of mass
    """
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def get_center_of_geometry(atoms):
    """
    Computes center of geometry.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of geometry
    """
    return atoms.arrays["positions"].mean(0)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database. Use together with SchNet4AIM.data.AtomsLoader to feed data
    to your model.

    Args:
        dbpath (str): path to directory containing database.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        available_properties (list, optional): complete set of physical properties
            that are contained in the database.
        load_only (list, optional): reduced set of properties to be loaded
        units (list, optional): definition of units for all available properties
        environment_provider (s4aim.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=s4aim.environment.SimpleEnvironmentProvider).
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
    """

    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        subset=None,
        available_properties=None,
        load_only=None,
        units=None,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        centering_function=get_center_of_mass,
    ):
        if not dbpath.endswith((".db",".json")) :
            raise AtomsDataError(
                "Invalid database path! Please make sure to add the file extension '.db' or '.json' to "
                "your dbpath."
            )
        # Check for incompatibilities with the db databases
        if (s4aim.DBmode == "db"):
           if subset is not None:
               raise AtomsDataError(
                   "The subset argument is deprecated and can not be used anymore! "
                   "Please use s4aim.data.partitioning.create_subset or "
                   "s4aim.data.AtomsDataSubset to build subsets."
               )

        # database
        self.dbpath = dbpath

        # Update the database if deprecated
        if (s4aim.DBmode == "db"):
           # check if database is deprecated:
           if self._is_deprecated():
               self._deprecation_update()

        self._load_only = load_only
        self._available_properties = self._get_available_properties(
            available_properties
        )

        if units is None:
            units = [1.0] * len(self.available_properties)
        self.units = dict(zip(self.available_properties, units))

        if len(units) != len(self.available_properties):
            raise AtomsDataError(
                "The length of available properties and units does not match!"
            )

        # environment
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centering_function = centering_function

    @property
    def available_properties(self):
        return self._available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.available_properties
        return self._load_only

    @property
    def atomref(self):
        return self.get_atomref(self.load_only)

    if (s4aim.DBmode == "db"):
       # metadata
       def get_metadata(self, key=None):
           """
           Returns an entry from the metadata dictionary of the ASE db.

           Args:
               key: Name of metadata entry. Return full dict if `None`.

           Returns:
               value: Value of metadata entry or full metadata dict, if key is `None`.

           """
           with ase.db.connect(self.dbpath) as conn:
               if key is None:
                   return conn.metadata
               if key in conn.metadata.keys():
                   return conn.metadata[key]
           return None

       def set_metadata(self, metadata=None, **kwargs):
           """
           Sets the metadata dictionary of the ASE db.

           Args:
               metadata (dict): dictionary of metadata for the ASE db
               kwargs: further key-value pairs for convenience
           """

           # merge all metadata
           if metadata is not None:
               kwargs.update(metadata)

           with ase.db.connect(self.dbpath) as conn:
               conn.metadata = kwargs

       def update_metadata(self, data):
           with ase.db.connect(self.dbpath) as conn:
               metadata = conn.metadata
           metadata.update(data)
           self.set_metadata(metadata)

       def get_atomref(self, properties):
           """
           Return multiple single atom reference values as a dictionary.

           Args:
               properties (list or str): Desired properties for which the atomrefs are
                   calculated.

           Returns:
               dict: atomic references
           """
           if type(properties) is not list:
               properties = [properties]
           return {p: self._get_atomref(p) for p in properties}

    # get atoms and properties
    def get_properties(self, idx, load_only=None):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index
            load_only (sequence or None): subset of available properties to load

        Returns:
            at (ase.Atoms): atoms object
            properties (dict): dictionary with molecular properties

        """
        # use all available properties if nothing is specified
        if load_only is None:
            load_only = self.available_properties

        if (s4aim.DBmode == "db"):
           # read from ase-database
           with ase.db.connect(self.dbpath) as conn:
               row = conn.get(idx + 1)
           at = row.toatoms()
        elif (s4aim.DBmode == "json"):
           # read from json-database
           with open(self.dbpath) as jsonfile:
                jsondat=json.load(jsonfile)
           jsonfile.close()
           # create an ase object
           at=json_2_ase_atoms(jsondat,idx)


        # extract properties
        properties = {}
        properties.clear() 
        if (s4aim.DBmode == "db"):
           for pname in load_only:
               properties[pname] = row.data[pname]
        elif (s4aim.DBmode == "json"):
           for pname in load_only:
               properties[pname] = jsondat[pname][idx]


        # extract/calculate structure
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            output=properties,
        )

        return at, properties

    if (s4aim.DBmode == "db"):
        def get_atoms(self, idx):
            """
            Return atoms of provided index.

            Args:
                idx (int): atoms index

            Returns:
                ase.Atoms: atoms data

            """
            with ase.db.connect(self.dbpath) as conn:
                row = conn.get(idx + 1)
            at = row.toatoms()
            return at

        # add systems
        def add_system(self, atoms, properties=dict(), key_value_pairs=dict()):
            """
            Add atoms data to the dataset.

            Args:
                atoms (ase.Atoms): system composition and geometry
                **properties: properties as key-value pairs. Keys have to match the
                    `available_properties` of the dataset.

            """
            with ase.db.connect(self.dbpath) as conn:
                self._add_system(conn, atoms, properties, key_value_pairs)

        def add_systems(self, atoms_list, property_list=None, key_value_pairs_list=None):
            """
            Add atoms data to the dataset.

            Args:
                atoms_list (list of ase.Atoms): system composition and geometry
                property_list (list): Properties as list of key-value pairs in the same
                    order as corresponding list of `atoms`.
                    Keys have to match the `available_properties` of the dataset.

            """
            # build empty dicts if property/kv_pairs list is None
            if property_list is None:
                property_list = [dict() for _ in range(len(atoms_list))]
            if key_value_pairs_list is None:
                key_value_pairs_list = [dict() for _ in range(len(atoms_list))]

            # write systems to database
            with ase.db.connect(self.dbpath) as conn:
                for at, prop, kv_pair in zip(
                    atoms_list, property_list, key_value_pairs_list
                ):
                    self._add_system(conn, at, prop, kv_pair)

        # deprecated
        def create_subset(self, subset):
            warnings.warn(
                "create_subset is deprecated! Please use "
                "s4aim.data.partitioning.create_subset.",
                DeprecationWarning,
            )
            from .partitioning import create_subset

            return create_subset(self, subset)

        # __functions__
        def __len__(self):
            with ase.db.connect(self.dbpath) as conn:
                return conn.count()

        def __getitem__(self, idx):
            _, properties = self.get_properties(idx, self.load_only)
            properties["_idx"] = np.array([idx], dtype=np.int)

            return torchify_dict(properties)

        # private methods
        def _add_system(self, conn, atoms, properties=dict(), key_value_pairs=dict()):
            """
            Write systems to the database. Floats, ints and np.ndarrays without dimension are transformed to np.ndarrays with dimension 1.

            """
            data = {}
            # add available properties to database
            for pname in self.available_properties:
                try:
                    data[pname] = properties[pname]
                except:
                    raise AtomsDataError("Required property missing:" + pname)

            # transform to np.ndarray
            data = numpyfy_dict(data)

            conn.write(atoms, data=data, key_value_pairs=key_value_pairs)

        def _get_atomref(self, property):
            """
            Returns single atom reference values for specified `property`.

            Args:
                property (str): property name

            Returns:
                list: list of atomrefs
            """
            labels = self.get_metadata("atref_labels")
            if labels is None:
                return None

            col = [i for i, l in enumerate(labels) if l == property]
            assert len(col) <= 1

            if len(col) == 1:
                col = col[0]
                atomref = np.array(self.get_metadata("atomrefs"))[:, col : col + 1]
            else:
                atomref = None

            return atomref
    elif (s4aim.DBmode == "json"):
         def __len__(self):
             with open(self.dbpath) as f:
                 dum = json.load(f)
             f.close()
             count=int(len(dum["name"]))
             return count

    def _get_available_properties(self, properties):
        """
        Get available properties from argument or database.

        Returns:
            (list): all properties of the dataset
        """
        if (s4aim.DBmode == "db"):
           # read database properties
           if os.path.exists(self.dbpath) and len(self) != 0:
               with ase.db.connect(self.dbpath) as conn:
                   atmsrw = conn.get(1)
                   db_properties = list(atmsrw.data.keys())
           else:
               db_properties = None

        elif (s4aim.DBmode == "json"):
           if os.path.exists(self.dbpath) and len(self.dbpath) != 0:
                with open(self.dbpath) as jsonfile:
                     jsondat=json.load(jsonfile)
                jsonfile.close()
                db_properties = list(jsondat.keys())
           else:
                db_properties = None

        # use the provided list
        if properties is not None:
            if db_properties is None or set(db_properties) == set(properties):
                return properties

            # raise error if available properties do not match database
            raise AtomsDataError(
                "The available_properties {} do not match the "
                "properties in the database {}!".format(properties, db_properties)
            )

        # return database properties
        if db_properties is not None:
            return db_properties

        raise AtomsDataError(
            "Please define available_properties or set db_path to an existing database!"
        )

    if (s4aim.DBmode == "db"):
       def _is_deprecated(self):
           """
           Check if database is deprecated.

           Returns:
               (bool): True if ase db is deprecated.
           """
           # check if db exists
           if not os.path.exists(self.dbpath):
               return False

           # get properties of first atom
           with ase.db.connect(self.dbpath) as conn:
               data = conn.get(1).data

           # check byte style deprecation
           if True in [pname.startswith("_dtype_") for pname in data.keys()]:
               return True
           # fallback for properties stored directly in the row
           if True in [type(val) != np.ndarray for val in data.values()]:
               return True

           return False

       def _deprecation_update(self):
           """
           Update deprecated database to a valid ase database.
           """
           warnings.warn(
               "The database is deprecated and will be updated automatically. "
               "The old database is moved to {}.deprecated!".format(self.dbpath)
           )

           # read old database
           (
               atoms_list,
               properties_list,
               key_value_pairs_list,
           ) = s4aim.utils.read_deprecated_database(self.dbpath)
           metadata = self.get_metadata()

           # move old database
           os.rename(self.dbpath, self.dbpath + ".deprecated")

           # write updated database
           self.set_metadata(metadata=metadata)
           with ase.db.connect(self.dbpath) as conn:
               for atoms, properties, key_value_pairs in tqdm(
                   zip(atoms_list, properties_list, key_value_pairs_list),
                   "Updating new database",
                   total=len(atoms_list),
               ):
                   conn.write(
                       atoms,
                       data=numpyfy_dict(properties),
                       key_value_pairs=key_value_pairs,
                   )

class AtomsDataSubset(Subset):
    r"""
    Subset of an atomistic dataset at specified indices.
    Arguments:
        dataset (torch.utils.data.Dataset): atomistic dataset
        indices (sequence): subset indices
    """

    def __init__(self, dataset, indices):
        super(AtomsDataSubset, self).__init__(dataset, indices)
        self._load_only = None

    @property
    def available_properties(self):
        return self.dataset.available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.dataset.load_only
        return self._load_only

    @property
    def atomref(self):
        return self.dataset.atomref

    def get_atomref(self, properties):
        return self.dataset.get_atomref(properties)

    def get_properties(self, idx, load_only=None):
        return self.dataset.get_properties(idx, load_only)

    def set_load_only(self, load_only):
        # check if properties are available
        for pname in load_only:
            if pname not in self.available_properties:
                raise AtomsDataError(
                    "The property '{}' is not an available property and can therefore "
                    "not be loaded!".format(pname)
                )

        # update load_only parameter
        self._load_only = list(load_only)

    # deprecated
    if (s4aim.DBmode == "db"):
       def create_subset(self, subset):
           warnings.warn(
               "create_subset is deprecated! Please use "
               "s4aim.data.partitioning.create_subset.",
               DeprecationWarning,
           )
           from .partitioning import create_subset
   
           return create_subset(self, subset)
   
    def __getitem__(self, idx):
        _, properties = self.get_properties(self.indices[idx], self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)


def _convert_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=None,
    output=None,
):
    """
    Helper function to convert ASE atoms object to SchNetPack input format.

    Args:
        atoms (ase.Atoms): Atoms object of molecule
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
        output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.

    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    inputs[Properties.Z] = atoms.numbers.astype(np.int)
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = positions

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    inputs[Properties.neighbors] = nbh_idx.astype(np.int)

    # Get cells
    inputs[Properties.cell] = np.array(atoms.cell.array, dtype=np.float32)
    inputs[Properties.cell_offset] = offsets.astype(np.float32)

    # If requested get neighbor lists for triples
    if collect_triples:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        inputs[Properties.neighbor_pairs_j] = nbh_idx_j.astype(np.int)
        inputs[Properties.neighbor_pairs_k] = nbh_idx_k.astype(np.int)

        inputs[Properties.neighbor_offsets_j] = offset_idx_j.astype(np.int)
        inputs[Properties.neighbor_offsets_k] = offset_idx_k.astype(np.int)

    return inputs


def torchify_dict(data):
    """
    Transform np.ndarrays to torch.tensors.

    """
    torch_properties = {}
    if (s4aim.DBmode =="json"):
       # Drop some of the variables that are not relevant
       data.pop("ele")                                                             
       data.pop("name")                                                            
       data.pop("pos")                                                             
       # Retrieve the properties in the database
       for pname, prop in data.items():
           prop=np.array(prop)                                                   
           if prop.dtype in [np.int, np.int32, np.int64]:
               torch_properties[pname] = torch.LongTensor(np.array(prop,float))
           elif prop.dtype in [np.float, np.float32, np.float64]:
               torch_properties[pname] = torch.FloatTensor(prop.copy())
    else:
       for pname, prop in data.items():
           if prop.dtype in [np.int, np.int32, np.int64]:
               torch_properties[pname] = torch.LongTensor(prop)
           elif prop.dtype in [np.float, np.float32, np.float64]:
               torch_properties[pname] = torch.FloatTensor(prop.copy())
           else:
               raise AtomsDataError(
                   "Invalid datatype {} for property {}!".format(type(prop), pname)
               )
    return torch_properties


def numpyfy_dict(data):
    """
    Transform floats, ints and dimensionless numpy in a dict to arrays to numpy arrays with dimenison.

    """
    for k, v in data.items():
        if type(v) in [int, float]:
            v = np.array([v])
        if v.shape == ():
            v = v[np.newaxis]
        data[k] = v
    return data


class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
    """

    def __init__(
        self,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=torch.device("cpu"),
    ):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples

        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms, self.environment_provider, self.collect_triples)
        inputs = torchify_dict(inputs)

        # Calculate masks
        inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
        mask = inputs[Properties.neighbors] >= 0
        inputs[Properties.neighbor_mask] = mask.float()
        inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
        )

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            inputs[key] = value.unsqueeze(0).to(self.device)

        return inputs
