"""Tests for the cylindrical lattice Python API."""

import h5py
import numpy as np
import lxml.etree as ET
import pytest

import openmc


def make_universes(count):
    return [openmc.Universe(cells=[openmc.Cell()]) for _ in range(count)]


def universe_ids(lattice):
    layers = [lattice.universes] if lattice.ndim == 2 else lattice.universes
    return [[[u.id for u in ring] for ring in layer] for layer in layers]


def make_lattice_2d(outer=None):
    universes = make_universes(5)
    return openmc.CylLattice(
        center=(1.5, -2.0), radial_edges=(0.5, 1.5, 3.0),
        sector_counts=(2, 3), azimuthal_offsets=(0.2, -0.1),
        universes=[[universes[0], universes[1]],
                   [universes[2], universes[3], universes[4]]],
        outer=outer)


def make_lattice_3d(outer=None):
    universes = make_universes(6)
    return openmc.CylLattice(
        center=(-0.5, 2.0), radial_edges=(0.0, 1.0, 2.5),
        sector_counts=(1, 2), azimuthal_offsets=(0.0, 0.3),
        z_edges=(-4.0, -1.0, 5.0),
        universes=[[[universes[0]], [universes[1], universes[2]]],
                   [[universes[3]], [universes[4], universes[5]]]],
        outer=outer)


def test_cyl_lattice_xml_roundtrip_2d_and_outer():
    outer = openmc.Universe(cells=[openmc.Cell()])
    lattice = make_lattice_2d(outer)
    xml = ET.Element('geometry')
    lattice.create_xml_subelement(xml)
    elem = xml.find('cylindrical_lattice')

    by_id = {u.id: u for u in [outer] + make_unique_universes(lattice)}
    restored = openmc.CylLattice.from_xml_element(elem, by_id.__getitem__)

    assert restored.center == lattice.center
    assert restored.radial_edges == lattice.radial_edges
    assert restored.sector_counts == lattice.sector_counts
    assert restored.azimuthal_offsets == lattice.azimuthal_offsets
    assert universe_ids(restored) == universe_ids(lattice)
    assert restored.outer is outer


def test_cyl_lattice_xml_roundtrip_3d():
    lattice = make_lattice_3d()
    xml = ET.Element('geometry')
    lattice.create_xml_subelement(xml)
    elem = xml.find('cylindrical_lattice')
    by_id = {u.id: u for u in make_unique_universes(lattice)}

    restored = openmc.CylLattice.from_xml_element(elem, by_id.__getitem__)

    assert restored.z_edges == lattice.z_edges
    assert universe_ids(restored) == universe_ids(lattice)


def make_unique_universes(lattice):
    layers = [lattice.universes] if lattice.ndim == 2 else lattice.universes
    result = []
    for layer in layers:
        for ring in layer:
            for universe in ring:
                if universe not in result:
                    result.append(universe)
    return result


@pytest.mark.parametrize('axial', [False, True])
def test_regular_equal_area_xml_roundtrip(axial):
    kwargs = dict(r_min=1.0, r_max=5.0, n_rings=3, n_sectors=2,
                  radial_spacing='equal_area', phi_offset=0.15,
                  center=(2.0, -1.0))
    if axial:
        kwargs.update(z_min=-3.0, z_max=9.0, n_axial=2)
        # Use exactly two universes per ring, with distinct identities per tile.
        layers = [[make_universes(6)[i:i + 2] for i in (0, 2, 4)] for _ in range(2)]
    else:
        flat = make_universes(6)
        layers = [flat[i:i + 2] for i in (0, 2, 4)]
    lattice = openmc.RegularCylLattice(**kwargs, universes=layers)
    xml = ET.Element('geometry')
    lattice.create_xml_subelement(xml)
    by_id = {u.id: u for u in make_unique_universes(lattice)}

    restored = openmc.CylLattice.from_xml_element(
        xml.find('cylindrical_lattice'), by_id.__getitem__)

    assert isinstance(restored, openmc.RegularCylLattice)
    assert restored.radial_spacing == 'equal_area'
    np.testing.assert_allclose(restored.radial_edges, np.sqrt(np.linspace(1.0, 25.0, 4)))
    assert universe_ids(restored) == universe_ids(lattice)
    if axial:
        np.testing.assert_allclose(restored.z_edges, (-3.0, 3.0, 9.0))


def write_hdf5_lattice(group, lattice):
    group.create_dataset('name', data=lattice.name.encode())
    group.create_dataset('center', data=lattice.center)
    group.create_dataset('radial_edges', data=lattice.radial_edges)
    group.create_dataset('sector_counts', data=lattice.sector_counts)
    group.create_dataset('azimuthal_offsets', data=lattice.azimuthal_offsets)
    if lattice.z_edges is not None:
        group.create_dataset('z_edges', data=lattice.z_edges)
    if isinstance(lattice, openmc.RegularCylLattice):
        group.create_dataset('regular', data=1)
        group.create_dataset('radial_spacing', data=lattice.radial_spacing.encode())
    else:
        group.create_dataset('regular', data=0)
    outer_id = -1 if lattice.outer is None else lattice.outer.id
    group.create_dataset('outer', data=outer_id)
    ids = []
    layers = [lattice.universes] if lattice.ndim == 2 else lattice.universes
    for layer in layers:
        for ring in layer:
            ids.extend(u.id for u in ring)
    group.create_dataset('universes', data=ids)


@pytest.mark.parametrize('make_lattice', [make_lattice_2d, make_lattice_3d])
def test_cyl_lattice_hdf5_roundtrip(tmp_path, make_lattice):
    outer = openmc.Universe(cells=[openmc.Cell()])
    lattice = make_lattice(outer)
    filename = tmp_path / 'cyl_lattice.h5'
    with h5py.File(filename, 'w') as fh:
        group = fh.create_group(f'lattice {lattice.id}')
        write_hdf5_lattice(group, lattice)
        restored = openmc.CylLattice.from_hdf5(
            group, {u.id: u for u in [outer] + make_unique_universes(lattice)})

    assert restored.radial_edges == lattice.radial_edges
    assert restored.sector_counts == lattice.sector_counts
    assert restored.azimuthal_offsets == lattice.azimuthal_offsets
    assert restored.z_edges == lattice.z_edges
    assert universe_ids(restored) == universe_ids(lattice)
    assert restored.outer is outer


def test_regular_equal_area_hdf5_roundtrip(tmp_path):
    lattice = openmc.RegularCylLattice(
        r_min=1.0, r_max=5.0, n_rings=3, n_sectors=2,
        radial_spacing='equal_area', universes=[make_universes(2) for _ in range(3)])
    with h5py.File(tmp_path / 'regular_cyl.h5', 'w') as fh:
        group = fh.create_group(f'lattice {lattice.id}')
        write_hdf5_lattice(group, lattice)
        restored = openmc.CylLattice.from_hdf5(
            group, {u.id: u for u in make_unique_universes(lattice)})
    assert isinstance(restored, openmc.RegularCylLattice)
    assert restored.radial_spacing == 'equal_area'
    np.testing.assert_allclose(restored.radial_edges, lattice.radial_edges)


def test_regular_cyl_lattice_hdf5_rejects_nonuniform_edges(tmp_path):
    lattice = openmc.RegularCylLattice(
        r_min=0.0, r_max=3.0, n_rings=3, n_sectors=2,
        universes=[make_universes(2) for _ in range(3)])
    with h5py.File(tmp_path / 'invalid_regular_cyl.h5', 'w') as fh:
        group = fh.create_group(f'lattice {lattice.id}')
        write_hdf5_lattice(group, lattice)
        edges = group['radial_edges'][()]
        edges[1] += 0.1
        group['radial_edges'][...] = edges
        with pytest.raises(ValueError, match='nonuniform coordinates'):
            openmc.CylLattice.from_hdf5(
                group, {u.id: u for u in make_unique_universes(lattice)})


def test_cyl_lattice_clone_3d_preserves_layout_and_clones_tiles():
    lattice = make_lattice_3d()
    clone = lattice.clone()

    assert clone.z_edges == lattice.z_edges
    assert [len(ring) for layer in clone.universes for ring in layer] == [
        len(ring) for layer in lattice.universes for ring in layer]
    assert clone.outer is None
    original_ids = {u.id for u in make_unique_universes(lattice)}
    cloned_ids = {u.id for u in make_unique_universes(clone)}
    assert len(cloned_ids) == len(original_ids)
    assert cloned_ids.isdisjoint(original_ids)
    for z, layer in enumerate(lattice.universes):
        for r, ring in enumerate(layer):
            for p, original in enumerate(ring):
                copied = clone.universes[z][r][p]
                assert copied is not original
                assert copied.id != original.id


def test_cyl_lattice_degenerate_discretization_clones_selected_materials():
    selected = openmc.Material(name='selected')
    selected.add_element('H', 1.0)
    unselected = openmc.Material(name='unselected')
    unselected.add_element('O', 1.0)
    cells = [openmc.Cell(fill=selected), openmc.Cell(fill=unselected)]
    shared = openmc.Universe(cells=cells)
    outer = openmc.Universe(cells=[openmc.Cell()])
    lattice = openmc.CylLattice(
        radial_edges=(0.0, 1.0), sector_counts=(2,),
        universes=[[shared, shared]], outer=outer)

    lattice.discretize(materials_to_clone=[selected])

    first, second = [lattice.get_universe((0, p)) for p in range(2)]
    assert first is not second
    assert first.cells[next(iter(first.cells))].fill is not second.cells[next(iter(second.cells))].fill
    selected_fills = [next(c for c in u.cells.values() if c.fill.name == 'selected').fill
                      for u in (first, second)]
    unselected_fills = [next(c for c in u.cells.values() if c.fill.name == 'unselected').fill
                        for u in (first, second)]
    assert selected_fills[0] is not selected_fills[1]
    assert unselected_fills[0] is unselected_fills[1]
    assert lattice.outer is outer


def test_cyl_lattice_lns_groups_equal_neighborhoods_only():
    a, b, c, d = make_universes(4)
    lattice = openmc.CylLattice(
        radial_edges=(0.0, 1.0, 2.0, 3.0), sector_counts=(2, 4, 4),
        azimuthal_offsets=(0.0, 0.0, 0.1),
        universes=[[a, a], [b, c, b, c], [d, d, d, d]])

    lattice.discretize(strategy='lns', key=lambda universe: universe.id)

    inner, middle, outer = lattice.universes
    # The two inner A tiles have the same angular/radial neighbors and merge.
    assert inner[0] is inner[1]
    # Middle tiles with different adjacent universes must remain distinct.
    assert middle[0] is middle[2]
    assert middle[1] is middle[3]
    assert middle[0] is not middle[1]
    # The offset outer ring still groups only matching neighbor sectors.
    assert outer[0] is outer[2]
    assert outer[1] is outer[3]
    assert outer[0] is not outer[1]
