/**
 * @file
 * @brief Utility functions.
 */

#ifndef ETTCM_UTILS_HPP
#define ETTCM_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ETTCM/assert.hpp"
#include "ETTCM/types.hpp"
#include "pontella.hpp"
#include "sepia.hpp"

namespace ETTCM
{
/**
 * @brief Computes the base-e exponential function.
 *
 * This function returns the value of e (the base of natural logarithm) raised
 * to the power of \p x.
 *
 * @param x Input value.
 *
 * @return Exponential value of \p x.
 */
inline double
compute_exp(const double x)
{
#ifdef ETTCM_FAST_EXP
  if (x < -600)
  {
    return 0;
  }
  union
  {
    double d;
    long l;
  } u;
  u.l = long(double((long(1) << 52) / 0.69314718056) * x +
             double((long(1) << 52) * 1023));
  return u.d;
#else
  return std::exp(x);
#endif
}

/**
 * @brief Computes the local window boundaries centered at the input
 * coordinates.
 *
 * @param x x-coordinate.
 * @param y y-coordinate.
 * @param width Context width.
 * @param height Context height.
 * @param x_radius Local x-radius.
 * @param y_radius Local y-radius.
 * @param[out] x_min Minimum local x-coordinate.
 * @param[out] x_max Maximum local x-coordinate.
 * @param[out] y_min Minimum local y-coordinate.
 * @param[out] y_max Maximum local y-coordinate.
 */
inline void
compute_local_window_boundaries(const uint16_t x, const uint16_t y,
                                const uint16_t width, const uint16_t height,
                                const uint16_t x_radius,
                                const uint16_t y_radius, uint16_t& x_min,
                                uint16_t& x_max, uint16_t& y_min,
                                uint16_t& y_max)
{
  x_min = std::max(static_cast<int>(x) - static_cast<int>(x_radius), 0);
  y_min = std::max(static_cast<int>(y) - static_cast<int>(y_radius), 0);
  x_max = std::min(static_cast<int>(x) + static_cast<int>(x_radius),
                   static_cast<int>(width - 1));
  y_max = std::min(static_cast<int>(y) + static_cast<int>(y_radius),
                   static_cast<int>(height - 1));
}

/**
 * @brief Extracts the argument value from a command.
 *
 * This function returns the argument value from a \p command given its \p name.
 * In case the argument \p name is not found, the default value is returned.
 *
 * @tparam T Type of the argument value.
 *
 * @param command Command line.
 * @param name Name of the argument.
 * @param default_argument Default argument value.
 *
 * @return Argument value if the argument \p name is found in the \p command.
 * @return Passed argument value if the argument \p name is not found in the \p
 * command.
 */
template <typename T>
T
extract_argument(pontella::command command, const std::string& name,
                 T default_argument)
{
  const auto name_and_argument = command.options.find(name);
  if (name_and_argument != command.options.end())
  {
    std::stringstream name_stream(name_and_argument->second);
    T argument;
    name_stream >> argument;
    return argument;
  }
  return default_argument;
}

/**
 * @brief Gets the file stream handle.
 *
 * @param filename Full path to the file.
 *
 * @return File stream handle.
 */
inline std::ifstream
filename_to_ifstream(const std::string& filename)
{
  auto stream = std::ifstream(filename, std::ifstream::in);
  if (!stream.good())
  {
    throw sepia::unreadable_file(filename);
  }
  return stream;
}

/**
 * @brief Reads the camera calibration parameters from a file.
 *
 * This function reads the camera calibration parameters from a file.
 * The calibration file should contain the camera intrinsics and distortion
 * parameters, as well as the context size in the following format:
 *
 * \f[f_{x}~f_{y}~c_{x}~c_{y}~d_{1}~d_{2}~d_{3}~d_{4}~d_{5}\f]
 *
 * \f[w~h,\f]
 *
 * where \f$w\f$ and \f$h\f$ are the width and height values of the context,
 * respectively.
 *
 * @tparam T Numerical type of the matrix elements.
 *
 * @param filename Name of the calibration file.
 * @param[out] width Width of the context \f$w\f$.
 * @param[out] height Height of the context \f$h\f$.
 * @param[out] intrinsics Camera intrinsics of the form
 * \f$\begin{bmatrix}
 *    f_{x} & 0 & c_{x} \\
 *    0 & f_{y} & c_{y} \\
 *    0 & 0 & 1
 * \end{bmatrix}\f$.
 * @param[out] distortion_parameters Camera distortion parameters of the form
 * \f$\begin{bmatrix}d_{1} & d_{2} & d_{3} & d_{4} & d_{5}\end{bmatrix}\f$.
 *
 * @return Returns true if the file exists and false otherwise.
 */
template <typename T>
void
read_calibration(const std::string& filename, uint16_t& width, uint16_t& height,
                 Matrix<T, 3, 3>& intrinsics,
                 Vector<T, 5>& distortion_parameters)
{
  std::ifstream fin(filename_to_ifstream(filename));

  // intrinsics
  intrinsics.setIdentity();
  fin >> intrinsics(0, 0);
  fin >> intrinsics(1, 1);
  fin >> intrinsics(0, 2);
  fin >> intrinsics(1, 2);

  // distortion parameters
  for (uint16_t i = 0; i < 5; ++i)
  {
    fin >> distortion_parameters(i);
  }

  // context size
  fin >> width;
  fin >> height;
}

/**
 * @brief Reads one value.
 *
 * @tparam T Type of the value.
 *
 * @param fin File stream handle.
 * @param[out] val Last value.
 */
template <typename T>
inline void
read_value(std::ifstream& fin, T& val)
{
  if (fin.eof())
  {
    throw sepia::end_of_file();
  }

  fin >> val;
}

/**
 * @brief Reads one value.
 *
 * @tparam T Type of the value.
 *
 * @param filename File name.
 * @param[out] val Last value.
 */
template <typename T>
inline void
read_value(const std::string& filename, T& val)
{
  std::ifstream fin(filename_to_ifstream(filename));

  read_value<T>(fin, val);
}

/**
 * @brief Converts coordinates from 2D array to 1D array.
 *
 * @param row Row index.
 * @param column Column index.
 * @param height Height of the original 2D array.
 *
 * @return 1D array coordinate.
 */
inline uint64_t
row_column_to_ind(const uint16_t row, const uint16_t column,
                  const uint16_t height)
{
  return row * height + column;
}

/**
 * @brief String format.
 *
 * @tparam Args Types of the remaining arguments.
 *
 * @param format Input string format.
 * @param args Remaining arguments.
 *
 * @return Formatted string.
 */
template <typename... Args>
std::string
string_format(const std::string& format, Args... args)
{
  const int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
                     1;  // extra space for '\0'
  if (size_s <= 0)
  {
    throw std::runtime_error("Error during string formatting.");
  }
  const size_t size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1);  // we don't want the '\0' inside
}
}  // namespace ETTCM

#endif  // ETTCM_UTILS_HPP
