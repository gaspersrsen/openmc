"""Cylindrical lattice Python API."""
from numbers import Integral
from math import floor, pi

import lxml.etree as ET
import numpy as np

import openmc
from ._xml import get_elem_list, get_text
from .lattice import Lattice


class CylLattice(Lattice):
    """Radial, azimuthal and optional axial universe partition.

    ``radial_edges`` and optional ``z_edges`` are strictly increasing in cm.
    The first radial edge may be positive, defining an annular lattice.
    ``sector_counts`` and ``azimuthal_offsets`` have one entry per ring;
    offsets are counterclockwise radians from +x. Universes are ordered
    ``[ring][sector]`` in 2-D and ``[z][ring][sector]`` in 3-D, from low
    radius, low angle and low z. Sector zero starts at the given offset.

    Like other OpenMC lattices, a tile translates its universe. Its x-y
    origin is the polar midpoint of the tile (the lattice center for a
    complete annulus); its z origin is the axial midpoint for a 3-D tile.
    An ``outer`` universe fills the region inside the inner radius, outside
    the outer radius, and outside the z range, when present.
    """

    def __init__(self, lattice_id=None, name='', *, center=(0., 0.),
                 radial_edges=None, sector_counts=None, azimuthal_offsets=None,
                 z_edges=None, universes=None, outer=None, r_min=None, r_max=None):
        super().__init__(lattice_id, name)
        self.center = tuple(float(x) for x in center)
        if len(self.center) != 2 or not np.all(np.isfinite(self.center)):
            raise ValueError('center must contain two finite coordinates')
        if radial_edges is None and r_max is not None:
            radial_edges = [0. if r_min is None else r_min, r_max]
        self.radial_edges = None if radial_edges is None else tuple(float(x) for x in radial_edges)
        self.sector_counts = None if sector_counts is None else tuple(sector_counts)
        self.azimuthal_offsets = (None if azimuthal_offsets is None else
                                  tuple(float(x) for x in azimuthal_offsets))
        self.z_edges = None if z_edges is None else tuple(float(x) for x in z_edges)
        if universes is not None:
            self._universes = universes
        if outer is not None:
            self.outer = outer
        if self.radial_edges is not None:
            if r_min is not None and not np.isclose(self.radial_edges[0], r_min):
                raise ValueError('r_min must match the first radial edge')
            if r_max is not None and not np.isclose(self.radial_edges[-1], r_max):
                raise ValueError('r_max must match the last radial edge')

    @property
    def universes(self):
        return self._universes

    @universes.setter
    def universes(self, value):
        self._universes = value

    @property
    def ndim(self):
        return 2 if self.z_edges is None else 3

    @property
    def r_min(self):
        return self.radial_edges[0]

    @property
    def r_max(self):
        return self.radial_edges[-1]

    @property
    def indices(self):
        self._validate()
        if self.ndim == 2:
            return [(r, p) for r, n in enumerate(self.sector_counts) for p in range(n)]
        return [(z, r, p) for z in range(len(self.z_edges)-1)
                for r, n in enumerate(self.sector_counts) for p in range(n)]

    @property
    def _natural_indices(self):
        if self.ndim == 2:
            return self.indices
        return [(r, p, z) for z, r, p in self.indices]

    def _validate(self):
        r = self.radial_edges
        if r is None or len(r) < 2 or not np.all(np.isfinite(r)) or r[0] < 0 or np.any(np.diff(r) <= 0):
            raise ValueError('radial_edges must be nonnegative and strictly increasing')
        n = self.sector_counts
        if n is None or len(n) != len(r)-1 or any(not isinstance(x, Integral) or isinstance(x, bool) or x < 1 for x in n):
            raise ValueError('sector_counts must contain one positive integer per ring')
        if self.azimuthal_offsets is None:
            self.azimuthal_offsets = (0.,)*len(n)
        if len(self.azimuthal_offsets) != len(n) or not np.all(np.isfinite(self.azimuthal_offsets)):
            raise ValueError('azimuthal_offsets must contain one finite angle per ring')
        if self.z_edges is not None and (len(self.z_edges) < 2 or not np.all(np.isfinite(self.z_edges)) or np.any(np.diff(self.z_edges) <= 0)):
            raise ValueError('z_edges must be finite and strictly increasing')
        layers = [self.universes] if self.ndim == 2 else self.universes
        if layers is None or len(layers) != (1 if self.ndim == 2 else len(self.z_edges)-1):
            raise ValueError('universes do not match the number of axial layers')
        for layer in layers:
            if len(layer) != len(n):
                raise ValueError('universes do not match the number of rings')
            for ring, count in zip(layer, n):
                if len(ring) != count or any(not isinstance(u, openmc.UniverseBase) for u in ring):
                    raise ValueError('each ring must contain its specified number of universes')

    def find_element(self, point):
        self._validate()
        x, y, z = point
        dx, dy = x-self.center[0], y-self.center[1]
        r = int(np.searchsorted(self.radial_edges, np.hypot(dx, dy), side='right')-1)
        if 0 <= r < len(self.sector_counts):
            phi = (np.arctan2(dy, dx)-self.azimuthal_offsets[r]) % (2*pi)
            p = min(int(floor(phi*self.sector_counts[r]/(2*pi))), self.sector_counts[r]-1)
        else:
            p = 0
        idx = (r, p) if self.ndim == 2 else (r, p, int(np.searchsorted(self.z_edges, z, side='right')-1))
        return idx, self.get_local_coordinates(point, idx)

    def get_local_coordinates(self, point, idx):
        if not self.is_valid_index(idx):
            return tuple(point)
        x, y, z = point
        r, p = idx[:2]
        if 0 <= r < len(self.sector_counts) and 0 <= p < self.sector_counts[r]:
            n = self.sector_counts[r]
            mid = 0. if n == 1 else (self.radial_edges[r]+self.radial_edges[r+1])/2
            phi = self.azimuthal_offsets[r] + (p+.5)*2*pi/n
            x -= self.center[0]+mid*np.cos(phi)
            y -= self.center[1]+mid*np.sin(phi)
        if self.ndim == 3 and 0 <= idx[2] < len(self.z_edges)-1:
            z -= (self.z_edges[idx[2]]+self.z_edges[idx[2]+1])/2
        return x, y, z

    def is_valid_index(self, idx):
        r, p = idx[:2]
        return (0 <= r < len(self.sector_counts) and 0 <= p < self.sector_counts[r] and
                (self.ndim == 2 or 0 <= idx[2] < len(self.z_edges)-1))

    def get_universe_index(self, idx):
        return idx if self.ndim == 2 else (idx[2], idx[0], idx[1])

    def discretize(self, strategy='degenerate', universes_to_ignore=(),
                   materials_to_clone=(), lattice_neighbors=(), key=lambda u: u.id):
        """Clone each tile universe, optionally cloning selected materials.

        Supports both 2-D and 3-D for ``degenerate``. The ``lns`` strategy
        supports 2-D and conservatively groups tiles with the same radial and
        azimuthal neighboring universe keys and overlapping angular spans.
        ``lattice_neighbors``, if given, contains the inner and outer exterior
        universes; otherwise ``outer`` is used for both. Outer is unchanged.
        """
        if strategy not in ('degenerate', 'lns'):
            raise ValueError("strategy must be 'degenerate' or 'lns'")
        if strategy == 'lns' and self.ndim != 2:
            raise NotImplementedError('Cylindrical LNS requires a 2-D lattice')
        if not callable(key):
            raise TypeError('key must be callable')
        lattice_neighbors = tuple(lattice_neighbors)
        if len(lattice_neighbors) not in (0, 2) or any(
                not isinstance(u, openmc.UniverseBase) for u in lattice_neighbors):
            raise ValueError('lattice_neighbors must be empty or contain inner and outer universes')
        ignored = set(universes_to_ignore)
        materials_to_clone = tuple(materials_to_clone)
        for u in ignored:
            if not isinstance(u, openmc.UniverseBase):
                raise TypeError('universes_to_ignore must contain universes')
        for m in materials_to_clone:
            if not isinstance(m, openmc.Material):
                raise TypeError('materials_to_clone must contain materials')
        selected = {m.id for m in materials_to_clone}
        self._validate()
        layers = ([[list(ring) for ring in self.universes]] if self.ndim == 2
                  else [[list(ring) for ring in layer] for layer in self.universes])
        original = [ring[:] for ring in layers[0]] if self.ndim == 2 else None
        patterns = {}

        def neighborhood(ir, ip):
            n = self.sector_counts[ir]
            width = 2*pi/n
            mid = self.azimuthal_offsets[ir] + (ip+.5)*width
            own = original[ir]
            def adjacent(other):
                if other < 0 or other >= len(original):
                    edge = (lattice_neighbors[0 if other < 0 else 1]
                            if lattice_neighbors else self.outer)
                    return (('outer', None if edge is None else key(edge)),)
                count = self.sector_counts[other]
                step = 2*pi/count
                result = []
                for index, neighbor in enumerate(original[other]):
                    relative = ((self.azimuthal_offsets[other]+(index+.5)*step-mid+pi) % (2*pi))-pi
                    for shift in (-2*pi, 0., 2*pi):
                        center = relative+shift
                        left = max(-width/2, center-step/2)
                        right = min(width/2, center+step/2)
                        if right-left > 1.e-12:
                            result.append((round(left, 12), round(right, 12), key(neighbor)))
                return tuple(sorted(result, key=lambda item: item[:2]))
            return (key(own[ip]), round(width, 12), key(own[(ip-1)%n]),
                    key(own[(ip+1)%n]), adjacent(ir-1), adjacent(ir+1))

        for layer in layers:
            for ir, ring in enumerate(layer):
                for iphi, universe in enumerate(ring):
                    if universe in ignored:
                        continue
                    pattern = neighborhood(ir, iphi) if strategy == 'lns' else None
                    if pattern is not None and pattern in patterns:
                        ring[iphi] = patterns[pattern]
                        continue
                    clone = universe.clone(clone_materials=False, clone_regions=False)
                    replacements = {}
                    for cell in clone.get_all_cells().values():
                        if cell.fill_type == 'distribmat':
                            raise ValueError('Lattice discretization does not support distributed materials')
                        if cell.fill_type == 'material' and cell.fill.id in selected:
                            material = cell.fill
                            if material.id not in replacements:
                                replacements[material.id] = material.clone()
                            cell.fill = replacements[material.id]
                    ring[iphi] = clone
                    if pattern is not None:
                        patterns[pattern] = clone
        self.universes = layers[0] if self.ndim == 2 else layers

    def place(self, *, region=None, translation=None, rotation=None,
              cell_id=None, name=''):
        """Place the lattice in a cell with an optional rigid 3D transform.

        ``translation`` and ``rotation`` use OpenMC's cell conventions.
        Rotation may be three Euler angles in degrees or a 3-by-3 rotation
        matrix. The returned cell's transform can be reassigned before export;
        separate placements of one lattice can have different transforms.
        The cell region is expressed in its parent coordinates.
        """
        cell = openmc.Cell(cell_id=cell_id, name=name, fill=self, region=region)
        if translation is not None:
            value = np.asarray(translation, dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError('translation must contain three finite coordinates')
            cell.translation = value
        if rotation is not None:
            value = np.asarray(rotation, dtype=float)
            if value.shape not in ((3,), (3, 3)) or not np.all(np.isfinite(value)):
                raise ValueError('rotation must contain three finite angles or a 3-by-3 matrix')
            if value.shape == (3, 3) and (not np.allclose(value @ value.T, np.eye(3), rtol=0., atol=1.e-10)
                                          or not np.isclose(np.linalg.det(value), 1., rtol=0., atol=1.e-10)):
                raise ValueError('rotation matrix must be orthonormal with determinant +1')
            cell.rotation = value
        return cell

    def create_xml_subelement(self, xml_element, memo=None):
        self._validate()
        if memo is None:
            memo = set()
        if self in memo:
            return
        memo.add(self)
        elem = ET.Element('cylindrical_lattice', id=str(self.id), name=self.name)
        if isinstance(self, RegularCylLattice):
            elem.set('regular', 'true')
            elem.set('radial_spacing', self.radial_spacing)
        for tag, data in (('center', self.center), ('radial_edges', self.radial_edges),
                          ('sector_counts', self.sector_counts), ('azimuthal_offsets', self.azimuthal_offsets),
                          ('z_edges', self.z_edges)):
            if data is not None:
                ET.SubElement(elem, tag).text = ' '.join(map(str, data))
        if self.outer is not None:
            ET.SubElement(elem, 'outer').text = str(self.outer.id)
            self.outer.create_xml_subelement(xml_element, memo)
        layers = [self.universes] if self.ndim == 2 else self.universes
        ids = []
        for layer in layers:
            for ring in layer:
                for universe in ring:
                    ids.append(str(universe.id))
                    universe.create_xml_subelement(xml_element, memo)
        ET.SubElement(elem, 'universes').text = ' '.join(ids)
        xml_element.append(elem)

    @classmethod
    def from_xml_element(cls, elem, get_universe):
        r = get_elem_list(elem, 'radial_edges', float)
        n = get_elem_list(elem, 'sector_counts', int)
        z = get_elem_list(elem, 'z_edges', float) if get_text(elem, 'z_edges') is not None else None
        center = get_elem_list(elem, 'center', float)
        offsets = get_elem_list(elem, 'azimuthal_offsets', float)
        if get_text(elem, 'regular') == 'true':
            lat = RegularCylLattice(int(get_text(elem, 'id')), get_text(elem, 'name'),
                center=center, r_min=r[0], r_max=r[-1], n_rings=len(n), n_sectors=n[0],
                phi_offset=offsets[0], radial_spacing=get_text(elem, 'radial_spacing') or 'radius',
                z_min=None if z is None else z[0],
                z_max=None if z is None else z[-1], n_axial=None if z is None else len(z)-1)
            if not (np.allclose(lat.radial_edges, r) and np.allclose(lat.azimuthal_offsets, offsets)
                    and tuple(lat.sector_counts) == tuple(n)
                    and (z is None or np.allclose(lat.z_edges, z))):
                raise ValueError('Regular cylindrical lattice has nonuniform coordinates')
        else:
            lat = cls(int(get_text(elem, 'id')), get_text(elem, 'name'), center=center,
                      radial_edges=r, sector_counts=n, azimuthal_offsets=offsets, z_edges=z)
        ids = iter(get_elem_list(elem, 'universes', int))
        layers = [[[get_universe(next(ids)) for _ in range(count)] for count in n]
                  for _ in range(1 if z is None else len(z)-1)]
        lat.universes = layers[0] if z is None else layers
        outer = get_text(elem, 'outer')
        if outer is not None:
            lat.outer = get_universe(int(outer))
        lat._validate()
        return lat

    @classmethod
    def from_hdf5(cls, group, universes):
        lattice_id = int(group.name.split()[-1])
        name = group['name'][()].decode() if 'name' in group else ''
        r, n = group['radial_edges'][()], group['sector_counts'][()]
        z = group['z_edges'][()] if 'z_edges' in group else None
        offsets = group['azimuthal_offsets'][()]
        if 'regular' in group and int(group['regular'][()]):
            lat = RegularCylLattice(lattice_id, name, center=group['center'][()],
                r_min=r[0], r_max=r[-1], n_rings=len(n), n_sectors=int(n[0]),
                phi_offset=offsets[0], radial_spacing=(group['radial_spacing'][()].decode()
                    if 'radial_spacing' in group else 'radius'),
                z_min=None if z is None else z[0],
                z_max=None if z is None else z[-1], n_axial=None if z is None else len(z)-1)
            if not (np.allclose(lat.radial_edges, r) and np.allclose(lat.azimuthal_offsets, offsets)
                    and tuple(lat.sector_counts) == tuple(n)
                    and (z is None or np.allclose(lat.z_edges, z))):
                raise ValueError('Regular cylindrical lattice has nonuniform coordinates')
        else:
            lat = cls(lattice_id, name, center=group['center'][()], radial_edges=r,
                      sector_counts=n, azimuthal_offsets=offsets, z_edges=z)
        values = iter(group['universes'][()])
        layers = [[[universes[int(next(values))] for _ in range(int(count))] for count in n]
                  for _ in range(1 if z is None else len(z)-1)]
        lat.universes = layers[0] if z is None else layers
        outer = int(group['outer'][()])
        if outer >= 0:
            lat.outer = universes[outer]
        lat._validate()
        return lat


class RegularCylLattice(CylLattice):
    """Cylindrical lattice with equal radial or equal-area, azimuthal and axial spacing.

    ``radial_spacing`` is ``'radius'`` for equal radial widths or
    ``'equal_area'`` for equal cross-sectional areas (equal volumes for each
    axial and azimuthal interval).
    """

    def __init__(self, lattice_id=None, name='', *, r_min=0., r_max, n_rings,
                 n_sectors, phi_offset=0., z_min=None, z_max=None, n_axial=None,
                 center=(0., 0.), universes=None, outer=None, radial_spacing='radius'):
        if not isinstance(n_rings, Integral) or isinstance(n_rings, bool) or n_rings < 1:
            raise ValueError('n_rings must be a positive integer')
        if not isinstance(n_sectors, Integral) or isinstance(n_sectors, bool) or n_sectors < 1:
            raise ValueError('n_sectors must be a positive integer')
        if (z_min is None, z_max is None, n_axial is None) not in ((True, True, True), (False, False, False)):
            raise ValueError('z_min, z_max and n_axial must be specified together')
        if n_axial is not None and (not isinstance(n_axial, Integral) or isinstance(n_axial, bool) or n_axial < 1):
            raise ValueError('n_axial must be a positive integer')
        if radial_spacing not in ('radius', 'equal_area'):
            raise ValueError("radial_spacing must be 'radius' or 'equal_area'")
        if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min < 0 or r_max <= r_min:
            raise ValueError('radii must be finite with 0 <= r_min < r_max')
        self.radial_spacing = radial_spacing
        edges = (np.linspace(r_min, r_max, n_rings+1) if radial_spacing == 'radius'
                 else np.sqrt(np.linspace(r_min*r_min, r_max*r_max, n_rings+1)))
        super().__init__(lattice_id, name, center=center, radial_edges=edges,
                         sector_counts=[n_sectors]*n_rings, azimuthal_offsets=[phi_offset]*n_rings,
                         z_edges=None if n_axial is None else np.linspace(z_min, z_max, n_axial+1),
                         universes=universes, outer=outer, r_min=r_min, r_max=r_max)
