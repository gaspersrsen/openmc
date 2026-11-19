#include "openmc/lattice.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/string_utils.h"
#include "openmc/vector.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace model {
std::unordered_map<int32_t, int32_t> lattice_map;
vector<unique_ptr<Lattice>> lattices;
} // namespace model

//==============================================================================
// Lattice implementation
//==============================================================================

Lattice::Lattice(pugi::xml_node lat_node)
{
  if (check_for_node(lat_node, "id")) {
    id_ = std::stoi(get_node_value(lat_node, "id"));
  } else {
    fatal_error("Must specify id of lattice in geometry XML file.");
  }

  if (check_for_node(lat_node, "name")) {
    name_ = get_node_value(lat_node, "name");
  }

  if (check_for_node(lat_node, "outer")) {
    outer_ = std::stoi(get_node_value(lat_node, "outer"));
  }
}

//==============================================================================

LatticeIter Lattice::begin()
{
  return LatticeIter(*this, 0);
}

LatticeIter Lattice::end()
{
  return LatticeIter(*this, universes_.size());
}

int32_t& Lattice::back()
{
  return universes_.back();
}

ReverseLatticeIter Lattice::rbegin()
{
  return ReverseLatticeIter(*this, universes_.size() - 1);
}

ReverseLatticeIter Lattice::rend()
{
  return ReverseLatticeIter(*this, -1);
}

//==============================================================================

void Lattice::adjust_indices()
{
  // Adjust the indices for the universes array.
  for (LatticeIter it = begin(); it != end(); ++it) {
    int uid = *it;
    auto search = model::universe_map.find(uid);
    if (search != model::universe_map.end()) {
      *it = search->second;
    } else {
      fatal_error(fmt::format(
        "Invalid universe number {} specified on lattice {}", uid, id_));
    }
  }

  // Adjust the index for the outer universe.
  if (outer_ != NO_OUTER_UNIVERSE) {
    auto search = model::universe_map.find(outer_);
    if (search != model::universe_map.end()) {
      outer_ = search->second;
    } else {
      fatal_error(fmt::format(
        "Invalid universe number {} specified on lattice {}", outer_, id_));
    }
  }
}

//==============================================================================

int32_t Lattice::fill_offset_table(int32_t target_univ_id, int map,
  std::unordered_map<int32_t, int32_t>& univ_count_memo)
{
  // If the offsets have already been determined for this "map", don't bother
  // recalculating all of them and just return the total offset. Note that the
  // offsets_ array doesn't actually include the offset accounting for the last
  // universe, so we get the before-last offset for the given map and then
  // explicitly add the count for the last universe.
  if (offsets_[map * universes_.size() + this->begin().indx_] != C_NONE) {
    int last_offset =
      offsets_[(map + 1) * universes_.size() - this->begin().indx_ - 1];
    int last_univ = this->back();
    return last_offset +
           count_universe_instances(last_univ, target_univ_id, univ_count_memo);
  }

  int32_t offset = 0;
  for (LatticeIter it = begin(); it != end(); ++it) {
    offsets_[map * universes_.size() + it.indx_] = offset;
    offset += count_universe_instances(*it, target_univ_id, univ_count_memo);
  }

  return offset;
}

//==============================================================================

void Lattice::to_hdf5(hid_t lattices_group) const
{
  // Make a group for the lattice.
  std::string group_name {"lattice "};
  group_name += std::to_string(id_);
  hid_t lat_group = create_group(lattices_group, group_name);

  // Write the name and outer universe.
  if (!name_.empty()) {
    write_string(lat_group, "name", name_, false);
  }

  if (outer_ != NO_OUTER_UNIVERSE) {
    int32_t outer_id = model::universes[outer_]->id_;
    write_dataset(lat_group, "outer", outer_id);
  } else {
    write_dataset(lat_group, "outer", outer_);
  }

  // Call subclass-overriden function to fill in other details.
  to_hdf5_inner(lat_group);

  close_group(lat_group);
}

//==============================================================================
// RectLattice implementation
//==============================================================================

RectLattice::RectLattice(pugi::xml_node lat_node) : Lattice {lat_node}
{
  type_ = LatticeType::rect;

  // Read the number of lattice cells in each dimension.
  std::string dimension_str {get_node_value(lat_node, "dimension")};
  vector<std::string> dimension_words {split(dimension_str)};
  if (dimension_words.size() == 2) {
    n_cells_[0] = std::stoi(dimension_words[0]);
    n_cells_[1] = std::stoi(dimension_words[1]);
    n_cells_[2] = 1;
    is_3d_ = false;
  } else if (dimension_words.size() == 3) {
    n_cells_[0] = std::stoi(dimension_words[0]);
    n_cells_[1] = std::stoi(dimension_words[1]);
    n_cells_[2] = std::stoi(dimension_words[2]);
    is_3d_ = true;
  } else {
    fatal_error("Rectangular lattice must be two or three dimensions.");
  }

  // Read the lattice lower-left location.
  std::string ll_str {get_node_value(lat_node, "lower_left")};
  vector<std::string> ll_words {split(ll_str)};
  if (ll_words.size() != dimension_words.size()) {
    fatal_error("Number of entries on <lower_left> must be the same as the "
                "number of entries on <dimension>.");
  }
  lower_left_[0] = stod(ll_words[0]);
  lower_left_[1] = stod(ll_words[1]);
  if (is_3d_) {
    lower_left_[2] = stod(ll_words[2]);
  }

  // Read the lattice pitches.
  std::string pitch_str {get_node_value(lat_node, "pitch")};
  vector<std::string> pitch_words {split(pitch_str)};
  if (pitch_words.size() != dimension_words.size()) {
    fatal_error("Number of entries on <pitch> must be the same as the "
                "number of entries on <dimension>.");
  }
  pitch_[0] = stod(pitch_words[0]);
  pitch_[1] = stod(pitch_words[1]);
  if (is_3d_) {
    pitch_[2] = stod(pitch_words[2]);
  }

  // Read the universes and make sure the correct number was specified.
  std::string univ_str {get_node_value(lat_node, "universes")};
  vector<std::string> univ_words {split(univ_str)};
  if (univ_words.size() != n_cells_[0] * n_cells_[1] * n_cells_[2]) {
    fatal_error(fmt::format(
      "Expected {} universes for a rectangular lattice of size {}x{}x{} but {} "
      "were specified.",
      n_cells_[0] * n_cells_[1] * n_cells_[2], n_cells_[0], n_cells_[1],
      n_cells_[2], univ_words.size()));
  }

  // Parse the universes.
  universes_.resize(n_cells_[0] * n_cells_[1] * n_cells_[2], C_NONE);
  for (int iz = 0; iz < n_cells_[2]; iz++) {
    for (int iy = n_cells_[1] - 1; iy > -1; iy--) {
      for (int ix = 0; ix < n_cells_[0]; ix++) {
        int indx1 = n_cells_[0] * n_cells_[1] * iz +
                    n_cells_[0] * (n_cells_[1] - iy - 1) + ix;
        int indx2 = n_cells_[0] * n_cells_[1] * iz + n_cells_[0] * iy + ix;
        universes_[indx1] = std::stoi(univ_words[indx2]);
      }
    }
  }
}

//==============================================================================

const int32_t& RectLattice::operator[](const array<int, 3>& i_xyz)
{
  return universes_[get_flat_index(i_xyz)];
}

//==============================================================================

bool RectLattice::are_valid_indices(const array<int, 3>& i_xyz) const
{
  return ((i_xyz[0] >= 0) && (i_xyz[0] < n_cells_[0]) && (i_xyz[1] >= 0) &&
          (i_xyz[1] < n_cells_[1]) && (i_xyz[2] >= 0) &&
          (i_xyz[2] < n_cells_[2]));
}

//==============================================================================

std::pair<double, array<int, 3>> RectLattice::distance(
  Position r, Direction u, const array<int, 3>& i_xyz) const
{
  // Get short aliases to the coordinates.
  double x = r.x;
  double y = r.y;
  double z = r.z;

  // Determine the oncoming edge.
  double x0 {copysign(0.5 * pitch_[0], u.x)};
  double y0 {copysign(0.5 * pitch_[1], u.y)};

  // Evaluate the distance to each oncoming edge independently. Comparing these
  // distances directly (rather than reconstructing the crossing position)
  // avoids the floating-point cancellation that occurs for large pitches.
  double dx = u.x != 0.0 ? (x0 - x) / u.x : INFTY;
  double dy = u.y != 0.0 ? (y0 - y) / u.y : INFTY;
  double dz = INFTY;
  if (is_3d_) {
    double z0 {copysign(0.5 * pitch_[2], u.z)};
    dz = u.z != 0.0 ? (z0 - z) / u.z : INFTY;
  }

  // The distance to the nearest lattice boundary is the smallest axial
  // distance.
  double d = std::min({dx, dy, dz});

  // Determine which lattice boundaries are being crossed. The axis attaining
  // the minimum is exactly equal to d, so at least one translation is always
  // set for a finite crossing; a near-equal second axis indicates a corner
  // crossing.
  array<int, 3> lattice_trans = {0, 0, 0};
  if (isclose(d, dx, FP_COINCIDENT, FP_PRECISION))
    lattice_trans[0] = copysign(1, u.x);
  if (isclose(d, dy, FP_COINCIDENT, FP_PRECISION))
    lattice_trans[1] = copysign(1, u.y);
  if (is_3d_ && isclose(d, dz, FP_COINCIDENT, FP_PRECISION))
    lattice_trans[2] = copysign(1, u.z);

  return {d, lattice_trans};
}

//==============================================================================

void RectLattice::get_indices(
  Position r, Direction u, array<int, 3>& result) const
{
  // Determine x index, accounting for coincidence
  double ix_ {(r.x - lower_left_.x) / pitch_.x};
  long ix_close {std::lround(ix_)};
  if (coincident(ix_, ix_close)) {
    result[0] = (u.x > 0) ? ix_close : ix_close - 1;
  } else {
    result[0] = std::floor(ix_);
  }

  // Determine y index, accounting for coincidence
  double iy_ {(r.y - lower_left_.y) / pitch_.y};
  long iy_close {std::lround(iy_)};
  if (coincident(iy_, iy_close)) {
    result[1] = (u.y > 0) ? iy_close : iy_close - 1;
  } else {
    result[1] = std::floor(iy_);
  }

  // Determine z index, accounting for coincidence
  result[2] = 0;
  if (is_3d_) {
    double iz_ {(r.z - lower_left_.z) / pitch_.z};
    long iz_close {std::lround(iz_)};
    if (coincident(iz_, iz_close)) {
      result[2] = (u.z > 0) ? iz_close : iz_close - 1;
    } else {
      result[2] = std::floor(iz_);
    }
  }
}

int RectLattice::get_flat_index(const array<int, 3>& i_xyz) const
{
  return n_cells_[0] * n_cells_[1] * i_xyz[2] + n_cells_[0] * i_xyz[1] +
         i_xyz[0];
}

//==============================================================================

Position RectLattice::get_local_position(
  Position r, const array<int, 3>& i_xyz) const
{
  r.x -= (lower_left_.x + (i_xyz[0] + 0.5) * pitch_.x);
  r.y -= (lower_left_.y + (i_xyz[1] + 0.5) * pitch_.y);
  if (is_3d_) {
    r.z -= (lower_left_.z + (i_xyz[2] + 0.5) * pitch_.z);
  }
  return r;
}

//==============================================================================

Direction RectLattice::get_normal(
  const array<int, 3>& i_xyz, bool& is_valid) const
{
  is_valid = false;
  Direction dir = {0.0, 0.0, 0.0};
  if ((std::abs(i_xyz[0]) == 1) && (i_xyz[1] == 0) && (i_xyz[2] == 0)) {
    is_valid = true;
    dir[0] = std::copysign(1.0, i_xyz[0]);
  } else if ((i_xyz[0] == 0) && (std::abs(i_xyz[1]) == 1) && (i_xyz[2] == 0)) {
    is_valid = true;
    dir[1] = std::copysign(1.0, i_xyz[1]);
  } else if ((i_xyz[0] == 0) && (i_xyz[1] == 0) && (std::abs(i_xyz[2]) == 1)) {
    is_valid = true;
    dir[2] = std::copysign(1.0, i_xyz[2]);
  }
  return dir;
}

//==============================================================================

int32_t& RectLattice::offset(int map, const array<int, 3>& i_xyz)
{
  return offsets_[n_cells_[0] * n_cells_[1] * n_cells_[2] * map +
                  n_cells_[0] * n_cells_[1] * i_xyz[2] +
                  n_cells_[0] * i_xyz[1] + i_xyz[0]];
}

//==============================================================================

int32_t RectLattice::offset(int map, int indx) const
{
  return offsets_[n_cells_[0] * n_cells_[1] * n_cells_[2] * map + indx];
}

//==============================================================================

std::string RectLattice::index_to_string(int indx) const
{
  int iz {indx / (n_cells_[0] * n_cells_[1])};
  int iy {(indx - n_cells_[0] * n_cells_[1] * iz) / n_cells_[0]};
  int ix {indx - n_cells_[0] * n_cells_[1] * iz - n_cells_[0] * iy};
  std::string out {std::to_string(ix)};
  out += ',';
  out += std::to_string(iy);
  if (is_3d_) {
    out += ',';
    out += std::to_string(iz);
  }
  return out;
}

//==============================================================================

void RectLattice::to_hdf5_inner(hid_t lat_group) const
{
  // Write basic lattice information.
  write_string(lat_group, "type", "rectangular", false);
  if (is_3d_) {
    write_dataset(lat_group, "pitch", pitch_);
    write_dataset(lat_group, "lower_left", lower_left_);
    write_dataset(lat_group, "dimension", n_cells_);
  } else {
    array<double, 2> pitch_short {{pitch_[0], pitch_[1]}};
    write_dataset(lat_group, "pitch", pitch_short);
    array<double, 2> ll_short {{lower_left_[0], lower_left_[1]}};
    write_dataset(lat_group, "lower_left", ll_short);
    array<int, 2> nc_short {{n_cells_[0], n_cells_[1]}};
    write_dataset(lat_group, "dimension", nc_short);
  }

  // Write the universe ids.  The convention here is to switch the ordering on
  // the y-axis to match the way universes are input in a text file.
  if (is_3d_) {
    hsize_t nx {static_cast<hsize_t>(n_cells_[0])};
    hsize_t ny {static_cast<hsize_t>(n_cells_[1])};
    hsize_t nz {static_cast<hsize_t>(n_cells_[2])};
    vector<int> out(nx * ny * nz);

    for (int m = 0; m < nz; m++) {
      for (int k = 0; k < ny; k++) {
        for (int j = 0; j < nx; j++) {
          int indx1 = nx * ny * m + nx * k + j;
          int indx2 = nx * ny * m + nx * (ny - k - 1) + j;
          out[indx2] = model::universes[universes_[indx1]]->id_;
        }
      }
    }

    hsize_t dims[3] {nz, ny, nx};
    write_int(lat_group, 3, dims, "universes", out.data(), false);

  } else {
    hsize_t nx {static_cast<hsize_t>(n_cells_[0])};
    hsize_t ny {static_cast<hsize_t>(n_cells_[1])};
    vector<int> out(nx * ny);

    for (int k = 0; k < ny; k++) {
      for (int j = 0; j < nx; j++) {
        int indx1 = nx * k + j;
        int indx2 = nx * (ny - k - 1) + j;
        out[indx2] = model::universes[universes_[indx1]]->id_;
      }
    }

    hsize_t dims[3] {1, ny, nx};
    write_int(lat_group, 3, dims, "universes", out.data(), false);
  }
}

//==============================================================================
// HexLattice implementation
//==============================================================================

HexLattice::HexLattice(pugi::xml_node lat_node) : Lattice {lat_node}
{
  type_ = LatticeType::hex;

  // Read the number of lattice cells in each dimension.
  n_rings_ = std::stoi(get_node_value(lat_node, "n_rings"));
  if (check_for_node(lat_node, "n_axial")) {
    n_axial_ = std::stoi(get_node_value(lat_node, "n_axial"));
    is_3d_ = true;
  } else {
    n_axial_ = 1;
    is_3d_ = false;
  }

  // Read the lattice orientation.  Default to 'y'.
  if (check_for_node(lat_node, "orientation")) {
    std::string orientation = get_node_value(lat_node, "orientation");
    if (orientation == "y") {
      orientation_ = Orientation::y;
    } else if (orientation == "x") {
      orientation_ = Orientation::x;
    } else {
      fatal_error("Unrecognized orientation '" + orientation +
                  "' for lattice " + std::to_string(id_));
    }
  } else {
    orientation_ = Orientation::y;
  }

  // Read the lattice center.
  std::string center_str {get_node_value(lat_node, "center")};
  vector<std::string> center_words {split(center_str)};
  if (is_3d_ && (center_words.size() != 3)) {
    fatal_error("A hexagonal lattice with <n_axial> must have <center> "
                "specified by 3 numbers.");
  } else if (!is_3d_ && center_words.size() != 2) {
    fatal_error("A hexagonal lattice without <n_axial> must have <center> "
                "specified by 2 numbers.");
  }
  center_[0] = stod(center_words[0]);
  center_[1] = stod(center_words[1]);
  if (is_3d_) {
    center_[2] = stod(center_words[2]);
  }

  // Read the lattice pitches.
  std::string pitch_str {get_node_value(lat_node, "pitch")};
  vector<std::string> pitch_words {split(pitch_str)};
  if (is_3d_ && (pitch_words.size() != 2)) {
    fatal_error("A hexagonal lattice with <n_axial> must have <pitch> "
                "specified by 2 numbers.");
  } else if (!is_3d_ && (pitch_words.size() != 1)) {
    fatal_error("A hexagonal lattice without <n_axial> must have <pitch> "
                "specified by 1 number.");
  }
  pitch_[0] = stod(pitch_words[0]);
  if (is_3d_) {
    pitch_[1] = stod(pitch_words[1]);
  }

  // Read the universes and make sure the correct number was specified.
  int n_univ = (3 * n_rings_ * n_rings_ - 3 * n_rings_ + 1) * n_axial_;
  std::string univ_str {get_node_value(lat_node, "universes")};
  vector<std::string> univ_words {split(univ_str)};
  if (univ_words.size() != n_univ) {
    fatal_error(fmt::format(
      "Expected {} universes for a hexagonal lattice with {} rings and {} "
      "axial levels but {} were specified.",
      n_univ, n_rings_, n_axial_, univ_words.size()));
  }

  // Parse the universes.
  // Universes in hexagonal lattices are stored in a manner that represents
  // a skewed coordinate system: (x, alpha) in case of 'y' orientation
  // and (alpha,y) in 'x' one rather than (x, y).  There is
  // no obvious, direct relationship between the order of universes in the
  // input and the order that they will be stored in the skewed array so
  // the following code walks a set of index values across the skewed array
  // in a manner that matches the input order.  Note that i_x = 0, i_a = 0
  // or i_a = 0, i_y = 0 corresponds to the center of the hexagonal lattice.
  universes_.resize((2 * n_rings_ - 1) * (2 * n_rings_ - 1) * n_axial_, C_NONE);
  if (orientation_ == Orientation::y) {
    fill_lattice_y(univ_words);
  } else {
    fill_lattice_x(univ_words);
  }
}

//==============================================================================

void HexLattice::fill_lattice_x(const vector<std::string>& univ_words)
{
  int input_index = 0;
  for (int m = 0; m < n_axial_; m++) {
    // Initialize lattice indecies.
    int i_a = -(n_rings_ - 1);
    int i_y = n_rings_ - 1;

    // Map upper region of hexagonal lattice which is found in the
    // first n_rings-1 rows of the input.
    for (int k = 0; k < n_rings_ - 1; k++) {

      // Iterate over the input columns.
      for (int j = 0; j < k + n_rings_; j++) {
        int indx = (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * m +
                   (2 * n_rings_ - 1) * (i_y + n_rings_ - 1) +
                   (i_a + n_rings_ - 1);
        universes_[indx] = std::stoi(univ_words[input_index]);
        input_index++;
        // Move to the next right neighbour cell
        i_a += 1;
      }

      // Return the lattice index to the start of the current row.
      i_a = -(n_rings_ - 1);
      i_y -= 1;
    }

    // Map the lower region from the centerline of cart to down side
    for (int k = 0; k < n_rings_; k++) {
      // Walk the index to the lower-right neighbor of the last row start.
      i_a = -(n_rings_ - 1) + k;

      // Iterate over the input columns.
      for (int j = 0; j < 2 * n_rings_ - k - 1; j++) {
        int indx = (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * m +
                   (2 * n_rings_ - 1) * (i_y + n_rings_ - 1) +
                   (i_a + n_rings_ - 1);
        universes_[indx] = std::stoi(univ_words[input_index]);
        input_index++;
        // Move to the next right neighbour cell
        i_a += 1;
      }

      // Return lattice index to start of current row.
      i_y -= 1;
    }
  }
}

//==============================================================================

void HexLattice::fill_lattice_y(const vector<std::string>& univ_words)
{
  int input_index = 0;
  for (int m = 0; m < n_axial_; m++) {
    // Initialize lattice indecies.
    int i_x = 1;
    int i_a = n_rings_ - 1;

    // Map upper triangular region of hexagonal lattice which is found in the
    // first n_rings-1 rows of the input.
    for (int k = 0; k < n_rings_ - 1; k++) {
      // Walk the index to lower-left neighbor of last row start.
      i_x -= 1;

      // Iterate over the input columns.
      for (int j = 0; j < k + 1; j++) {
        int indx = (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * m +
                   (2 * n_rings_ - 1) * (i_a + n_rings_ - 1) +
                   (i_x + n_rings_ - 1);
        universes_[indx] = std::stoi(univ_words[input_index]);
        input_index++;
        // Walk the index to the right neighbor (which is not adjacent).
        i_x += 2;
        i_a -= 1;
      }

      // Return the lattice index to the start of the current row.
      i_x -= 2 * (k + 1);
      i_a += (k + 1);
    }

    // Map the middle square region of the hexagonal lattice which is found in
    // the next 2*n_rings-1 rows of the input.
    for (int k = 0; k < 2 * n_rings_ - 1; k++) {
      if ((k % 2) == 0) {
        // Walk the index to the lower-left neighbor of the last row start.
        i_x -= 1;
      } else {
        // Walk the index to the lower-right neighbor of the last row start.
        i_x += 1;
        i_a -= 1;
      }

      // Iterate over the input columns.
      for (int j = 0; j < n_rings_ - (k % 2); j++) {
        int indx = (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * m +
                   (2 * n_rings_ - 1) * (i_a + n_rings_ - 1) +
                   (i_x + n_rings_ - 1);
        universes_[indx] = std::stoi(univ_words[input_index]);
        input_index++;
        // Walk the index to the right neighbor (which is not adjacent).
        i_x += 2;
        i_a -= 1;
      }

      // Return the lattice index to the start of the current row.
      i_x -= 2 * (n_rings_ - (k % 2));
      i_a += n_rings_ - (k % 2);
    }

    // Map the lower triangular region of the hexagonal lattice.
    for (int k = 0; k < n_rings_ - 1; k++) {
      // Walk the index to the lower-right neighbor of the last row start.
      i_x += 1;
      i_a -= 1;

      // Iterate over the input columns.
      for (int j = 0; j < n_rings_ - k - 1; j++) {
        int indx = (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * m +
                   (2 * n_rings_ - 1) * (i_a + n_rings_ - 1) +
                   (i_x + n_rings_ - 1);
        universes_[indx] = std::stoi(univ_words[input_index]);
        input_index++;
        // Walk the index to the right neighbor (which is not adjacent).
        i_x += 2;
        i_a -= 1;
      }

      // Return lattice index to start of current row.
      i_x -= 2 * (n_rings_ - k - 1);
      i_a += n_rings_ - k - 1;
    }
  }
}

//==============================================================================

const int32_t& HexLattice::operator[](const array<int, 3>& i_xyz)
{
  return universes_[get_flat_index(i_xyz)];
}

//==============================================================================

// The HexLattice iterators need their own versions b/c the universes array is
// "square", meaning that it is allocated with entries that are intentionally
// left empty. As such, the iterator indices need to skip the empty entries to
// get cell instances and geometry paths correct. See the image in the Theory
// and Methodology section on "Hexagonal Lattice Indexing" for a visual of where
// the empty positions are.
LatticeIter HexLattice::begin()
{
  return LatticeIter(*this, n_rings_ - 1);
}

ReverseLatticeIter HexLattice::rbegin()
{
  return ReverseLatticeIter(*this, universes_.size() - n_rings_);
}

int32_t& HexLattice::back()
{
  return universes_[universes_.size() - n_rings_];
}

LatticeIter HexLattice::end()
{
  return LatticeIter(*this, universes_.size() - n_rings_ + 1);
}

ReverseLatticeIter HexLattice::rend()
{
  return ReverseLatticeIter(*this, n_rings_ - 2);
}

//==============================================================================

bool HexLattice::are_valid_indices(const array<int, 3>& i_xyz) const
{
  // Check if (x, alpha, z) indices are valid, accounting for number of rings
  return ((i_xyz[0] >= 0) && (i_xyz[1] >= 0) && (i_xyz[2] >= 0) &&
          (i_xyz[0] < 2 * n_rings_ - 1) && (i_xyz[1] < 2 * n_rings_ - 1) &&
          (i_xyz[0] + i_xyz[1] > n_rings_ - 2) &&
          (i_xyz[0] + i_xyz[1] < 3 * n_rings_ - 2) && (i_xyz[2] < n_axial_));
}

//==============================================================================

std::pair<double, array<int, 3>> HexLattice::distance(
  Position r, Direction u, const array<int, 3>& i_xyz) const
{
  // Short description of the direction vectors used here.  The beta, gamma, and
  // delta vectors point towards the flat sides of each hexagonal tile.
  // Y - orientation:
  //   basis0 = (1, 0)
  //   basis1 = (-1/sqrt(3), 1)   = +120 degrees from basis0
  //   beta   = (sqrt(3)/2, 1/2)  = +30 degrees from basis0
  //   gamma  = (sqrt(3)/2, -1/2) = -60 degrees from beta
  //   delta  = (0, 1)            = +60 degrees from beta
  // X - orientation:
  //   basis0 = (1/sqrt(3), -1)
  //   basis1 = (0, 1)            = +120 degrees from basis0
  //   beta   = (1, 0)            = +30 degrees from basis0
  //   gamma  = (1/2, -sqrt(3)/2) = -60 degrees from beta
  //   delta  = (1/2, sqrt(3)/2)  = +60 degrees from beta
  // The z-axis is considered separately.
  double beta_dir;
  double gamma_dir;
  double delta_dir;
  if (orientation_ == Orientation::y) {
    beta_dir = u.x * std::sqrt(3.0) / 2.0 + u.y / 2.0;
    gamma_dir = u.x * std::sqrt(3.0) / 2.0 - u.y / 2.0;
    delta_dir = u.y;
  } else {
    beta_dir = u.x;
    gamma_dir = u.x / 2.0 - u.y * std::sqrt(3.0) / 2.0;
    delta_dir = u.x / 2.0 + u.y * std::sqrt(3.0) / 2.0;
  }

  // Note that hexagonal lattice distance calculations are performed
  // using the particle's coordinates relative to the neighbor lattice
  // cells, not relative to the particle's current cell.  This is done
  // because there is significant disagreement between neighboring cells
  // on where the lattice boundary is due to finite precision issues.

  // beta direction
  double d {INFTY};
  array<int, 3> lattice_trans;
  double edge = -copysign(0.5 * pitch_[0], beta_dir); // Oncoming edge
  Position r_t;
  if (beta_dir > 0) {
    const array<int, 3> i_xyz_t {i_xyz[0] + 1, i_xyz[1], i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  } else {
    const array<int, 3> i_xyz_t {i_xyz[0] - 1, i_xyz[1], i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  }
  double beta;
  if (orientation_ == Orientation::y) {
    beta = r_t.x * std::sqrt(3.0) / 2.0 + r_t.y / 2.0;
  } else {
    beta = r_t.x;
  }
  if ((std::abs(beta - edge) > FP_PRECISION) && beta_dir != 0) {
    d = (edge - beta) / beta_dir;
    if (beta_dir > 0) {
      lattice_trans = {1, 0, 0};
    } else {
      lattice_trans = {-1, 0, 0};
    }
  }

  // gamma direction
  edge = -copysign(0.5 * pitch_[0], gamma_dir);
  if (gamma_dir > 0) {
    const array<int, 3> i_xyz_t {i_xyz[0] + 1, i_xyz[1] - 1, i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  } else {
    const array<int, 3> i_xyz_t {i_xyz[0] - 1, i_xyz[1] + 1, i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  }
  double gamma;
  if (orientation_ == Orientation::y) {
    gamma = r_t.x * std::sqrt(3.0) / 2.0 - r_t.y / 2.0;
  } else {
    gamma = r_t.x / 2.0 - r_t.y * std::sqrt(3.0) / 2.0;
  }
  if ((std::abs(gamma - edge) > FP_PRECISION) && gamma_dir != 0) {
    double this_d = (edge - gamma) / gamma_dir;
    if (this_d < d) {
      if (gamma_dir > 0) {
        lattice_trans = {1, -1, 0};
      } else {
        lattice_trans = {-1, 1, 0};
      }
      d = this_d;
    }
  }

  // delta direction
  edge = -copysign(0.5 * pitch_[0], delta_dir);
  if (delta_dir > 0) {
    const array<int, 3> i_xyz_t {i_xyz[0], i_xyz[1] + 1, i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  } else {
    const array<int, 3> i_xyz_t {i_xyz[0], i_xyz[1] - 1, i_xyz[2]};
    r_t = get_local_position(r, i_xyz_t);
  }
  double delta;
  if (orientation_ == Orientation::y) {
    delta = r_t.y;
  } else {
    delta = r_t.x / 2.0 + r_t.y * std::sqrt(3.0) / 2.0;
  }
  if ((std::abs(delta - edge) > FP_PRECISION) && delta_dir != 0) {
    double this_d = (edge - delta) / delta_dir;
    if (this_d < d) {
      if (delta_dir > 0) {
        lattice_trans = {0, 1, 0};
      } else {
        lattice_trans = {0, -1, 0};
      }
      d = this_d;
    }
  }

  // Top and bottom sides
  if (is_3d_) {
    double z = r.z;
    double z0 {copysign(0.5 * pitch_[1], u.z)};
    if ((std::abs(z - z0) > FP_PRECISION) && u.z != 0) {
      double this_d = (z0 - z) / u.z;
      if (this_d < d) {
        d = this_d;
        if (u.z > 0) {
          lattice_trans = {0, 0, 1};
        } else {
          lattice_trans = {0, 0, -1};
        }
        d = this_d;
      }
    }
  }

  return {d, lattice_trans};
}

//==============================================================================

void HexLattice::get_indices(
  Position r, Direction u, array<int, 3>& result) const
{
  // Offset the xyz by the lattice center.
  Position r_o {r.x - center_.x, r.y - center_.y, r.z};
  if (is_3d_) {
    r_o.z -= center_.z;
  }

  // Index the z direction, accounting for coincidence
  result[2] = 0;
  if (is_3d_) {
    double iz_ {r_o.z / pitch_[1] + 0.5 * n_axial_};
    long iz_close {std::lround(iz_)};
    if (coincident(iz_, iz_close)) {
      result[2] = (u.z > 0) ? iz_close : iz_close - 1;
    } else {
      result[2] = std::floor(iz_);
    }
  }

  if (orientation_ == Orientation::y) {
    // Convert coordinates into skewed bases.  The (x, alpha) basis is used to
    // find the index of the global coordinates to within 4 cells.
    double alpha = r_o.y - r_o.x / std::sqrt(3.0);
    result[0] = std::floor(r_o.x / (0.5 * std::sqrt(3.0) * pitch_[0]));
    result[1] = std::floor(alpha / pitch_[0]);
  } else {
    // Convert coordinates into skewed bases.  The (alpha, y) basis is used to
    // find the index of the global coordinates to within 4 cells.
    double alpha = r_o.y - r_o.x * std::sqrt(3.0);
    result[0] = std::floor(-alpha / (std::sqrt(3.0) * pitch_[0]));
    result[1] = std::floor(r_o.y / (0.5 * std::sqrt(3.0) * pitch_[0]));
  }

  // Add offset to indices (the center cell is (i1, i2) = (0, 0) but
  // the array is offset so that the indices never go below 0).
  result[0] += n_rings_ - 1;
  result[1] += n_rings_ - 1;

  // Calculate the (squared) distance between the particle and the centers of
  // the four possible cells.  Regular hexagonal tiles form a Voronoi
  // tessellation so the xyz should be in the hexagonal cell that it is closest
  // to the center of.  This method is used over a method that uses the
  // remainders of the floor divisions above because it provides better finite
  // precision performance.  Squared distances are used because they are more
  // computationally efficient than normal distances.

  // COINCIDENCE CHECK
  // if a distance to center, d, is within the coincidence tolerance of the
  // current minimum distance, d_min, the particle is on an edge or vertex.
  // In this case, the dot product of the position vector and direction vector
  // for the current indices, dp, and the dot product for the currently selected
  // indices, dp_min, are compared. The cell which the particle is moving into
  // is kept (i.e. the cell with the lowest dot product as the vectors will be
  // completely opposed if the particle is moving directly toward the center of
  // the cell).
  int i1_chg;
  int i2_chg;
  double d_min {INFTY};
  double dp_min {INFTY};
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      // get local coordinates
      const array<int, 3> i_xyz {result[0] + j, result[1] + i, 0};
      Position r_t = get_local_position(r, i_xyz);
      // calculate distance
      double d = r_t.x * r_t.x + r_t.y * r_t.y;
      // check for coincidence. Because the numerical error incurred
      // in hex geometry is higher than other geometries, the relative
      // coincidence is checked here so that coincidence is successfully
      // detected on large hex lattice with particles far from the origin
      // which have rounding errors larger than the FP_COINCIDENT thresdhold.
      bool on_boundary = coincident(1.0, d_min / d);
      if (d < d_min || on_boundary) {
        // normalize r_t and find dot product
        r_t /= std::sqrt(d);
        double dp = u.x * r_t.x + u.y * r_t.y;
        // do not update values if particle is on a
        // boundary and not moving into this cell
        if (on_boundary && dp > dp_min)
          continue;
        // update values
        d_min = d;
        i1_chg = j;
        i2_chg = i;
        dp_min = dp;
      }
    }
  }

  // update outgoing indices
  result[0] += i1_chg;
  result[1] += i2_chg;
}

int HexLattice::get_flat_index(const array<int, 3>& i_xyz) const
{
  return (2 * n_rings_ - 1) * (2 * n_rings_ - 1) * i_xyz[2] +
         (2 * n_rings_ - 1) * i_xyz[1] + i_xyz[0];
}

//==============================================================================

Position HexLattice::get_local_position(
  Position r, const array<int, 3>& i_xyz) const
{
  if (orientation_ == Orientation::y) {
    // x_l = x_g - (center + pitch_x*cos(30)*index_x)
    r.x -=
      center_.x + std::sqrt(3.0) / 2.0 * (i_xyz[0] - n_rings_ + 1) * pitch_[0];
    // y_l = y_g - (center + pitch_x*index_x + pitch_y*sin(30)*index_y)
    r.y -= (center_.y + (i_xyz[1] - n_rings_ + 1) * pitch_[0] +
            (i_xyz[0] - n_rings_ + 1) * pitch_[0] / 2.0);
  } else {
    // x_l = x_g - (center + pitch_x*index_a + pitch_y*sin(30)*index_y)
    r.x -= (center_.x + (i_xyz[0] - n_rings_ + 1) * pitch_[0] +
            (i_xyz[1] - n_rings_ + 1) * pitch_[0] / 2.0);
    // y_l = y_g - (center + pitch_y*cos(30)*index_y)
    r.y -=
      center_.y + std::sqrt(3.0) / 2.0 * (i_xyz[1] - n_rings_ + 1) * pitch_[0];
  }

  if (is_3d_) {
    r.z -= center_.z - (0.5 * n_axial_ - i_xyz[2] - 0.5) * pitch_[1];
  }

  return r;
}

//==============================================================================

Direction HexLattice::get_normal(
  const array<int, 3>& i_xyz, bool& is_valid) const
{
  // Short description of the direction vectors used here.  The beta, gamma, and
  // delta vectors point towards the flat sides of each hexagonal tile.
  // Y - orientation:
  //   basis0 = (1, 0)
  //   basis1 = (-1/sqrt(3), 1)   = +120 degrees from basis0
  //   beta   = (sqrt(3)/2, 1/2)  = +30 degrees from basis0
  //   gamma  = (sqrt(3)/2, -1/2) = -60 degrees from beta
  //   delta  = (0, 1)            = +60 degrees from beta
  // X - orientation:
  //   basis0 = (1/sqrt(3), -1)
  //   basis1 = (0, 1)            = +120 degrees from basis0
  //   beta   = (1, 0)            = +30 degrees from basis0
  //   gamma  = (1/2, -sqrt(3)/2) = -60 degrees from beta
  //   delta  = (1/2, sqrt(3)/2)  = +60 degrees from beta

  is_valid = false;
  Direction dir = {0.0, 0.0, 0.0};
  if ((i_xyz[0] == 0) && (i_xyz[1] == 0) && (std::abs(i_xyz[2]) == 1)) {
    is_valid = true;
    dir[2] = std::copysign(1.0, i_xyz[2]);
  } else if ((i_xyz[2] == 0) &&
             std::max({std::abs(i_xyz[0]), std::abs(i_xyz[1]),
               std::abs(i_xyz[0] + i_xyz[1])}) == 1) {
    is_valid = true;
    // beta direction
    if ((i_xyz[0] == 1) && (i_xyz[1] == 0)) {
      if (orientation_ == Orientation::y) {
        dir[0] = 0.5 * std::sqrt(3.0);
        dir[1] = 0.5;
      } else {
        dir[0] = 1.0;
        dir[1] = 0.0;
      }
    } else if ((i_xyz[0] == -1) && (i_xyz[1] == 0)) {
      if (orientation_ == Orientation::y) {
        dir[0] = -0.5 * std::sqrt(3.0);
        dir[1] = -0.5;
      } else {
        dir[0] = -1.0;
        dir[1] = 0.0;
      }
      // gamma direction
    } else if ((i_xyz[0] == 1) && (i_xyz[1] == -1)) {
      if (orientation_ == Orientation::y) {
        dir[0] = 0.5 * std::sqrt(3.0);
        dir[1] = -0.5;
      } else {
        dir[0] = 0.5;
        dir[1] = -0.5 * std::sqrt(3.0);
      }
    } else if ((i_xyz[0] == -1) && (i_xyz[1] == 1)) {
      if (orientation_ == Orientation::y) {
        dir[0] = -0.5 * std::sqrt(3.0);
        dir[1] = 0.5;
      } else {
        dir[0] = -0.5;
        dir[1] = 0.5 * std::sqrt(3.0);
      }
      // delta direction
    } else if ((i_xyz[0] == 0) && (i_xyz[1] == 1)) {
      if (orientation_ == Orientation::y) {
        dir[0] = 0.0;
        dir[1] = 1.0;
      } else {
        dir[0] = 0.5;
        dir[1] = 0.5 * std::sqrt(3.0);
      }
    } else if ((i_xyz[0] == 0) && (i_xyz[1] == -1)) {
      if (orientation_ == Orientation::y) {
        dir[0] = 0.0;
        dir[1] = -1.0;
      } else {
        dir[0] = -0.5;
        dir[1] = -0.5 * std::sqrt(3.0);
      }
    }
  }
  return dir;
}

//==============================================================================

bool HexLattice::is_valid_index(int indx) const
{
  int nx {2 * n_rings_ - 1};
  int ny {2 * n_rings_ - 1};
  int iz = indx / (nx * ny);
  int iy = (indx - nx * ny * iz) / nx;
  int ix = indx - nx * ny * iz - nx * iy;
  array<int, 3> i_xyz {ix, iy, iz};
  return are_valid_indices(i_xyz);
}

//==============================================================================

int32_t& HexLattice::offset(int map, const array<int, 3>& i_xyz)
{
  int nx {2 * n_rings_ - 1};
  int ny {2 * n_rings_ - 1};
  int nz {n_axial_};
  return offsets_[nx * ny * nz * map + nx * ny * i_xyz[2] + nx * i_xyz[1] +
                  i_xyz[0]];
}

int32_t HexLattice::offset(int map, int indx) const
{
  int nx {2 * n_rings_ - 1};
  int ny {2 * n_rings_ - 1};
  int nz {n_axial_};
  return offsets_[nx * ny * nz * map + indx];
}

//==============================================================================

std::string HexLattice::index_to_string(int indx) const
{
  int nx {2 * n_rings_ - 1};
  int ny {2 * n_rings_ - 1};
  int iz {indx / (nx * ny)};
  int iy {(indx - nx * ny * iz) / nx};
  int ix {indx - nx * ny * iz - nx * iy};
  std::string out {std::to_string(ix - n_rings_ + 1)};
  out += ',';
  out += std::to_string(iy - n_rings_ + 1);
  if (is_3d_) {
    out += ',';
    out += std::to_string(iz);
  }
  return out;
}

//==============================================================================

void HexLattice::to_hdf5_inner(hid_t lat_group) const
{
  // Write basic lattice information.
  write_string(lat_group, "type", "hexagonal", false);
  write_dataset(lat_group, "n_rings", n_rings_);
  write_dataset(lat_group, "n_axial", n_axial_);
  if (orientation_ == Orientation::y) {
    write_string(lat_group, "orientation", "y", false);
  } else {
    write_string(lat_group, "orientation", "x", false);
  }
  if (is_3d_) {
    write_dataset(lat_group, "pitch", pitch_);
    write_dataset(lat_group, "center", center_);
  } else {
    array<double, 1> pitch_short {{pitch_[0]}};
    write_dataset(lat_group, "pitch", pitch_short);
    array<double, 2> center_short {{center_[0], center_[1]}};
    write_dataset(lat_group, "center", center_short);
  }

  // Write the universe ids.
  hsize_t nx {static_cast<hsize_t>(2 * n_rings_ - 1)};
  hsize_t ny {static_cast<hsize_t>(2 * n_rings_ - 1)};
  hsize_t nz {static_cast<hsize_t>(n_axial_)};
  vector<int> out(nx * ny * nz);

  for (int m = 0; m < nz; m++) {
    for (int k = 0; k < ny; k++) {
      for (int j = 0; j < nx; j++) {
        int indx = nx * ny * m + nx * k + j;
        if (j + k < n_rings_ - 1) {
          // This array position is never used; put a -1 to indicate this.
          out[indx] = -1;
        } else if (j + k > 3 * n_rings_ - 3) {
          // This array position is never used; put a -1 to indicate this.
          out[indx] = -1;
        } else {
          out[indx] = model::universes[universes_[indx]]->id_;
        }
      }
    }
  }

  hsize_t dims[3] {nz, ny, nx};
  write_int(lat_group, 3, dims, "universes", out.data(), false);
}

//==============================================================================
// Non-method functions
//==============================================================================

CylLattice::CylLattice(pugi::xml_node node) : Lattice {node}
{
  type_ = LatticeType::cylindrical;
  regular_ = check_for_node(node, "regular") && get_node_value_bool(node, "regular");
  auto center = get_node_array<double>(node, "center");
  radial_edges_ = get_node_array<double>(node, "radial_edges");
  sector_counts_ = get_node_array<int>(node, "sector_counts");
  azimuthal_offsets_ = get_node_array<double>(node, "azimuthal_offsets");
  is_3d_ = check_for_node(node, "z_edges");
  if (is_3d_) z_edges_ = get_node_array<double>(node, "z_edges");
  auto strictly_increasing = [](const vector<double>& a) {
    return a.size() >= 2 && std::all_of(a.begin(), a.end(), [](double x) { return std::isfinite(x); }) &&
           std::adjacent_find(a.begin(), a.end(), std::greater_equal<double>()) == a.end();
  };
  if (center.size() != 2 || !std::isfinite(center[0]) || !std::isfinite(center[1]) ||
      !strictly_increasing(radial_edges_) || radial_edges_.front() < 0. ||
      sector_counts_.size()+1 != radial_edges_.size() ||
      azimuthal_offsets_.size() != sector_counts_.size() ||
      (is_3d_ && !strictly_increasing(z_edges_))) {
    fatal_error("Invalid cylindrical lattice boundaries or dimensions");
  }
  center_.x = center[0];
  center_.y = center[1];
  n_axial_ = is_3d_ ? static_cast<int>(z_edges_.size())-1 : 1;
  if (regular_) {
    auto spacing = check_for_node(node, "radial_spacing") ?
      get_node_value(node, "radial_spacing") : "radius";
    if (spacing != "radius" && spacing != "equal_area")
      fatal_error("Invalid regular cylindrical lattice radial spacing");
    equal_area_ = spacing == "equal_area";
    double start = equal_area_ ? radial_edges_.front()*radial_edges_.front() : radial_edges_.front();
    double end = equal_area_ ? radial_edges_.back()*radial_edges_.back() : radial_edges_.back();
    double pitch = (end-start)/sector_counts_.size();
    radial_inverse_pitch_ = 1./pitch;
    for (int i = 0; i < radial_edges_.size(); ++i) {
      double value = equal_area_ ? radial_edges_[i]*radial_edges_[i] : radial_edges_[i];
      if (std::abs(value-(start+i*pitch)) > 1.e-10*std::max(1., std::abs(end)) ||
          (i < sector_counts_.size() &&
           (sector_counts_[i] != sector_counts_[0] ||
            std::abs(azimuthal_offsets_[i]-azimuthal_offsets_[0]) > 1.e-12)))
        fatal_error("Regular cylindrical lattice has nonuniform boundaries or sectors");
    }
    if (is_3d_) {
      double pitch_z = (z_edges_.back()-z_edges_.front())/n_axial_;
      axial_inverse_pitch_ = 1./pitch_z;
      for (int i = 0; i < z_edges_.size(); ++i)
        if (std::abs(z_edges_[i]-(z_edges_.front()+i*pitch_z)) >
            1.e-10*std::max(1., std::abs(z_edges_.back()-z_edges_.front())))
          fatal_error("Regular cylindrical lattice has nonuniform axial boundaries");
    }
  }
  tiles_per_layer_ = 0;
  for (int i = 0; i < sector_counts_.size(); ++i) {
    if (sector_counts_[i] < 1 || !std::isfinite(azimuthal_offsets_[i]))
      fatal_error("Invalid cylindrical lattice sectors or offsets");
    ring_offsets_.push_back(tiles_per_layer_);
    tiles_per_layer_ += sector_counts_[i];
  }
  auto words = get_node_array<int>(node, "universes");
  if (words.size() != static_cast<size_t>(n_axial_ * tiles_per_layer_))
    fatal_error("Incorrect number of cylindrical lattice universes");
  universes_ = words;
}

const int32_t& CylLattice::operator[](const array<int, 3>& index)
{
  return universes_[get_flat_index(index)];
}

bool CylLattice::are_valid_indices(const array<int, 3>& i) const
{
  return i[0] >= 0 && i[0] < sector_counts_.size() && i[1] >= 0 &&
    i[1] < sector_counts_[i[0]] && i[2] >= 0 && i[2] < n_axial_;
}

int CylLattice::get_flat_index(const array<int, 3>& i) const
{
  return i[2]*tiles_per_layer_ + ring_offsets_[i[0]] + i[1];
}

void CylLattice::get_indices(Position r, Direction u, array<int, 3>& out) const
{
  const double x = r.x-center_.x, y = r.y-center_.y;
  const double rad = std::hypot(x, y);
  int ring;
  if (regular_) {
    if (rad < radial_edges_.front()) ring = -1;
    else if (rad >= radial_edges_.back()) ring = sector_counts_.size();
    else {
      double value = equal_area_ ? rad*rad : rad;
      double start = equal_area_ ? radial_edges_.front()*radial_edges_.front() : radial_edges_.front();
      ring = std::min(static_cast<int>((value-start)*radial_inverse_pitch_),
                      static_cast<int>(sector_counts_.size())-1);
    }
  } else ring = std::upper_bound(radial_edges_.begin(), radial_edges_.end(), rad)-radial_edges_.begin()-1;
  int r_lo = regular_ ? std::max(0, ring) : 0;
  int r_hi = regular_ ? std::min(static_cast<int>(radial_edges_.size())-1, ring+1) : radial_edges_.size()-1;
  for (int i = r_lo; i <= r_hi; ++i) {
    double radial_tolerance =
      64.0*FP_COINCIDENT*std::max(1., radial_edges_[i]);
    if (i > 0) {
      radial_tolerance = std::min(
        radial_tolerance, 0.25*(radial_edges_[i]-radial_edges_[i-1]));
    }
    if (i+1 < radial_edges_.size()) {
      radial_tolerance = std::min(
        radial_tolerance, 0.25*(radial_edges_[i+1]-radial_edges_[i]));
    }
    // Coordinate roundoff during a surface crossing can put an inward-going
    // particle slightly beyond the radial edge. Allow a modest multiple of
    // the base coincidence tolerance so it still selects the adjacent ring.
    // Cap it by neighboring ring widths to keep the boundary bands disjoint.
    if (std::abs(rad-radial_edges_[i]) <= radial_tolerance) {
      const double dr = x*u.x+y*u.y;
      ring = dr < 0. ? i-1 : i;
      break;
    }
  }
  out[0] = ring;
  out[1] = 0;
  if (ring >= 0 && ring < sector_counts_.size() && sector_counts_[ring] > 1) {
    double angle = rad <= FP_COINCIDENT ? std::atan2(u.y, u.x) : std::atan2(y, x);
    double turn = std::fmod(angle-azimuthal_offsets_[ring], 2*M_PI);
    if (turn < 0.) turn += 2*M_PI;
    double sector = turn*sector_counts_[ring]/(2*M_PI);
    double nearest = std::round(sector);
    int phi = std::floor(sector);
    if (std::abs(sector-nearest) < FP_COINCIDENT) {
      double angular_direction = x*u.y-y*u.x;
      phi = static_cast<int>(nearest) - (angular_direction < 0. ? 1 : 0);
    }
    int n = sector_counts_[ring];
    out[1] = ((phi % n)+n)%n;
  }
  out[2] = 0;
  if (is_3d_) {
    int axial;
    if (regular_) {
      if (r.z < z_edges_.front()) axial = -1;
      else if (r.z >= z_edges_.back()) axial = n_axial_;
      else axial = std::min(static_cast<int>((r.z-z_edges_.front())*axial_inverse_pitch_), n_axial_-1);
    } else axial = std::upper_bound(z_edges_.begin(), z_edges_.end(), r.z)-z_edges_.begin()-1;
    int z_lo = regular_ ? std::max(0, axial) : 0;
    int z_hi = regular_ ? std::min(static_cast<int>(z_edges_.size())-1, axial+1) : z_edges_.size()-1;
    for (int i = z_lo; i <= z_hi; ++i) {
      if (std::abs(r.z-z_edges_[i]) <= FP_COINCIDENT*std::max(1., std::abs(z_edges_[i]))) {
        axial = u.z < 0. ? i-1 : i;
        break;
      }
    }
    out[2] = axial;
  }
}

Position CylLattice::get_local_position(Position r, const array<int, 3>& i) const
{
  if (!are_valid_indices(i)) return r;
  if (i[0] >= 0 && i[0] < sector_counts_.size() && i[1] >= 0 && i[1] < sector_counts_[i[0]]) {
    int n = sector_counts_[i[0]];
    double radius = n == 1 ? 0. : (radial_edges_[i[0]]+radial_edges_[i[0]+1])/2;
    double phi = azimuthal_offsets_[i[0]] + (i[1]+.5)*2*M_PI/n;
    r.x -= center_.x+radius*std::cos(phi);
    r.y -= center_.y+radius*std::sin(phi);
  }
  if (is_3d_ && i[2] >= 0 && i[2] < n_axial_)
    r.z -= (z_edges_[i[2]]+z_edges_[i[2]+1])/2;
  return r;
}

std::pair<double, array<int, 3>> CylLattice::distance(
  Position r, Direction u, const array<int, 3>& index) const
{
  const double x = r.x-center_.x, y = r.y-center_.y;
  const double angular_rate = x*u.y-y*u.x;
  double best_distance = INFTY;
  array<int, 3> best_delta {};

  // A sector crossing can use its known +/-1 tile transition directly unless
  // it is coincident with a radial or axial boundary. In that corner case,
  // retain the full lookup so all crossed indices are resolved together.
  auto on_radial_boundary = [&](double radius) {
    const int n_rings = sector_counts_.size();
    int first_edge;
    int last_edge;
    if (index[0] < 0) {
      first_edge = last_edge = 0;
    } else if (index[0] >= n_rings) {
      first_edge = last_edge = n_rings;
    } else {
      first_edge = index[0];
      last_edge = index[0] + 1;
    }

    for (int i = first_edge; i <= last_edge; ++i) {
      double tolerance = 64.0*FP_COINCIDENT*
                         std::max(1., radial_edges_[i]);
      if (i > 0) {
        tolerance = std::min(
          tolerance, 0.25*(radial_edges_[i]-radial_edges_[i-1]));
      }
      if (i+1 < radial_edges_.size()) {
        tolerance = std::min(
          tolerance, 0.25*(radial_edges_[i+1]-radial_edges_[i]));
      }
      if (std::abs(radius-radial_edges_[i]) <= tolerance) return true;
    }
    return false;
  };

  auto on_axial_boundary = [&](double z) {
    if (!is_3d_) return false;
    int first_edge;
    int last_edge;
    if (index[2] < 0) {
      first_edge = last_edge = 0;
    } else if (index[2] >= n_axial_) {
      first_edge = last_edge = n_axial_;
    } else {
      first_edge = index[2];
      last_edge = index[2] + 1;
    }
    for (int i = first_edge; i <= last_edge; ++i) {
      const double tolerance = FP_COINCIDENT*
                              std::max(1., std::abs(z_edges_[i]));
      if (std::abs(z-z_edges_[i]) <= tolerance) return true;
    }
    return false;
  };

  // Only evaluate crossings that lie ahead and that actually change tile.
  // Radial and axial candidates still use get_indices to preserve variable
  // sector layouts and exact corner handling. An isolated sector crossing
  // already identifies its adjacent tile, so avoid a complete point lookup.
  auto consider = [&](double d, int azimuthal_step) {
    if (!std::isfinite(d) || d <= FP_COINCIDENT || d >= best_distance) return;
    array<int, 3> delta {};
    const Position hit = r + d*u;
    const double hit_x = hit.x-center_.x;
    const double hit_y = hit.y-center_.y;
    if (azimuthal_step != 0 &&
        !on_radial_boundary(std::hypot(hit_x, hit_y)) &&
        !on_axial_boundary(hit.z)) {
      delta[1] = azimuthal_step;
    } else {
      array<int, 3> next;
      get_indices(hit, u, next);
      delta = {{next[0]-index[0], next[1]-index[1], next[2]-index[2]}};
    }
    // Sector indices wrap, but lattice translations must move by one adjacent
    // tile rather than by the raw difference between 0 and n-1.
    if (delta[0] == 0 && delta[1] != 0 && angular_rate != 0.)
      delta[1] = angular_rate > 0. ? 1 : -1;
    if (delta[0] || delta[1] || delta[2]) {
      best_distance = d;
      best_delta = delta;
    }
  };

  // Along a straight flight x*u_y-y*u_x is constant, so its sign selects
  // which of this sector's two azimuthal faces can be reached next.
  if (index[0] >= 0 && index[0] < static_cast<int>(sector_counts_.size())) {
    const int n = sector_counts_[index[0]];
    if (n > 1 && index[1] >= 0 && index[1] < n && angular_rate != 0.) {
      const int side = angular_rate > 0. ? 1 : 0;
      const double phi = azimuthal_offsets_[index[0]] +
                         (index[1]+side)*2.*M_PI/n;
      const double cx = std::cos(phi), cy = std::sin(phi);
      const double denominator = cx*u.y-cy*u.x;
      if (std::abs(denominator) >= FP_COINCIDENT) {
        const double d = -(cx*y-cy*x)/denominator;
        if (d > FP_COINCIDENT && d < best_distance &&
            cx*(x+d*u.x)+cy*(y+d*u.y) >= -FP_COINCIDENT) {
          consider(d, angular_rate > 0. ? 1 : -1);
        }
      }
    }
  }

  // Test both radial faces of the current band (or the single exterior
  // interface). Keep only roots crossing in the direction that leaves this
  // radial index; the other root is either a tangency or a later return.
  const double a = u.x*u.x + u.y*u.y;
  if (a > 0.) {
    auto circle = [&](double radius, double crossing_direction) {
      if (radius <= 0.) return;
      const double b = x*u.x + y*u.y;
      const double c = x*x + y*y - radius*radius;
      const double discriminant = b*b - a*c;
      if (discriminant < 0.) return;
      const double root = std::sqrt(std::max(0., discriminant));
      const double roots[2] = {(-b-root)/a, (-b+root)/a};
      for (double d : roots) {
        if (!std::isfinite(d) || d <= FP_COINCIDENT || d >= best_distance) continue;
        const double radial_rate = (x+d*u.x)*u.x + (y+d*u.y)*u.y;
        if (crossing_direction*radial_rate > 0.) consider(d, 0);
      }
    };

    if (index[0] < 0) {
      circle(radial_edges_.front(), +1.);
    } else if (index[0] >= static_cast<int>(sector_counts_.size())) {
      circle(radial_edges_.back(), -1.);
    } else {
      circle(radial_edges_[index[0]], -1.);
      circle(radial_edges_[index[0]+1], +1.);
    }
  }

  // For an in-lattice axial index, motion selects exactly one of its two
  // planes. Outside the stack, only the entry plane in the flight direction
  // can be reached.
  if (is_3d_ && u.z != 0.) {
    double z_boundary = 0.;
    bool has_boundary = true;
    if (index[2] < 0) {
      if (u.z > 0.) z_boundary = z_edges_.front();
      else has_boundary = false;
    } else if (index[2] >= n_axial_) {
      if (u.z < 0.) z_boundary = z_edges_.back();
      else has_boundary = false;
    } else if (u.z < 0.) {
      z_boundary = z_edges_[index[2]];
    } else {
      z_boundary = z_edges_[index[2]+1];
    }
    if (has_boundary) consider((z_boundary-r.z)/u.z, 0);
  }

  return {best_distance, best_delta};
}

Direction CylLattice::get_normal(const array<int, 3>& index, bool& is_valid) const
{
  // Curved/azimuthal normals depend on position. Current lattice-normal API
  // has no position; do not claim a wrong surface-current normal.
  is_valid = index[0] == 0 && index[1] == 0 && std::abs(index[2]) == 1;
  return {0., 0., is_valid ? std::copysign(1., index[2]) : 0.};
}

Direction CylLattice::get_normal(Position r, Direction u,
  const array<int, 3>& index,
  const array<int, 3>& delta, bool& is_valid) const
{
  // Orient the normal from the old tile toward the tile selected by delta.
  // Radial and azimuthal faces need the crossing position to define their
  // normal; axial faces retain the ordinary Cartesian normal. As for
  // rectangular/hexagonal lattices, a crossing that changes two independent
  // geometric coordinates is a corner and has no unique face normal.
  if ((delta[0] != 0 && delta[2] != 0) ||
      (delta[0] == 0 && delta[1] != 0 && delta[2] != 0)) {
    is_valid = false;
    return {};
  }
  is_valid = true;
  if (delta[0] != 0) {
    double x = r.x - center_.x;
    double y = r.y - center_.y;
    double radius = std::hypot(x, y);
    if (radius <= FP_COINCIDENT) {
      is_valid = false;
      return {};
    }
    // A ring transition can also change the sector index solely because the
    // destination ring uses a different offset/count. Treat it as a true
    // radial/azimuthal corner only when the hit lies on an old-ring sector
    // ray and the trajectory has a tangential component.
    int ring = index[0];
    if (delta[1] != 0 && ring >= 0 &&
        ring < static_cast<int>(sector_counts_.size()) &&
        sector_counts_[ring] > 1) {
      int n = sector_counts_[ring];
      double angle = std::atan2(y, x);
      double step = 2. * M_PI / n;
      double lo = azimuthal_offsets_[ring] + index[1] * step;
      double hi = lo + step;
      double tol = FP_COINCIDENT * std::max(1., radius);
      bool on_sector_ray = std::abs(std::remainder(angle-lo, 2.*M_PI)) < tol ||
        std::abs(std::remainder(angle-hi, 2.*M_PI)) < tol;
      double tangential = -y*u.x + x*u.y;
      if (on_sector_ray && std::abs(tangential) > FP_COINCIDENT) {
        is_valid = false;
        return {};
      }
    }
    double sign = delta[0] > 0 ? 1. : -1.;
    return {sign*x/radius, sign*y/radius, 0.};
  }
  if (delta[1] != 0) {
    int ring = index[0];
    if (ring < 0 || ring >= static_cast<int>(sector_counts_.size()) ||
        sector_counts_[ring] <= 1) {
      is_valid = false;
      return {};
    }
    int n = sector_counts_[ring];
    double angle = std::atan2(r.y-center_.y, r.x-center_.x);
    double lo = azimuthal_offsets_[ring] + index[1]*2.*M_PI/n;
    double hi = lo + 2.*M_PI/n;
    double phi = std::abs(std::remainder(angle-lo, 2.*M_PI)) <
        std::abs(std::remainder(angle-hi, 2.*M_PI)) ? lo : hi;
    double nx = -std::sin(phi), ny = std::cos(phi);
    // The normal points toward the destination sector. This also resolves
    // the n=2 case, where either modular index difference is ambiguous.
    double sign = u.x*nx + u.y*ny >= 0. ? 1. : -1.;
    return {sign*nx, sign*ny, 0.};
  }
  if (delta[2] != 0) return {0., 0., std::copysign(1., delta[2])};
  is_valid = false;
  return {};
}

int32_t& CylLattice::offset(int map, const array<int, 3>& index)
{
  return offsets_[map*universes_.size()+get_flat_index(index)];
}

int32_t CylLattice::offset(int map, int indx) const
{
  return offsets_[map*universes_.size()+indx];
}

std::string CylLattice::index_to_string(int indx) const
{
  int z = indx/tiles_per_layer_;
  int tile = indx%tiles_per_layer_;
  int ring = std::upper_bound(ring_offsets_.begin(), ring_offsets_.end(), tile)-ring_offsets_.begin()-1;
  return fmt::format("{},{},{}", ring, tile-ring_offsets_[ring], z);
}

void CylLattice::to_hdf5_inner(hid_t group) const
{
  write_string(group, "type", "cylindrical", false);
  write_dataset(group, "regular", static_cast<int>(regular_));
  if (regular_) write_string(group, "radial_spacing", equal_area_ ? "equal_area" : "radius", false);
  array<double, 2> center {{center_.x, center_.y}};
  write_dataset(group, "center", center);
  write_dataset(group, "radial_edges", radial_edges_);
  write_dataset(group, "sector_counts", sector_counts_);
  write_dataset(group, "azimuthal_offsets", azimuthal_offsets_);
  if (is_3d_) write_dataset(group, "z_edges", z_edges_);
  vector<int> values;
  for (auto universe : universes_) values.push_back(model::universes[universe]->id_);
  write_dataset(group, "universes", values);
}

void read_lattices(pugi::xml_node node)
{
  for (pugi::xml_node lat_node : node.children("lattice")) {
    model::lattices.push_back(make_unique<RectLattice>(lat_node));
  }
  for (pugi::xml_node lat_node : node.children("hex_lattice")) {
    model::lattices.push_back(make_unique<HexLattice>(lat_node));
  }
  for (pugi::xml_node lat_node : node.children("cylindrical_lattice")) {
    model::lattices.push_back(make_unique<CylLattice>(lat_node));
  }

  // Fill the lattice map.
  for (int i_lat = 0; i_lat < model::lattices.size(); i_lat++) {
    int id = model::lattices[i_lat]->id_;
    auto in_map = model::lattice_map.find(id);
    if (in_map == model::lattice_map.end()) {
      model::lattice_map[id] = i_lat;
    } else {
      fatal_error(
        fmt::format("Two or more lattices use the same unique ID: {}", id));
    }
  }
}

} // namespace openmc
