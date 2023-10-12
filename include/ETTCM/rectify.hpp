/**
 * @file
 * @brief Rectify implementation.
 */

#ifndef ETTCM_RECTIFY_HPP
#define ETTCM_RECTIFY_HPP

#include <cstdint>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "ETTCM/assert.hpp"
#include "ETTCM/types.hpp"
#include "ETTCM/utils.hpp"

namespace ETTCM
{
/**
 * @brief Compute the rectify maps from a calibration file.
 *
 * This class computes the rectify maps from a calibration file.
 * The calibration file should contain the camera intrinsics and distortion
 * parameters, as well as the context size.
 * \see read_calibration() for the exact format.
 */
class Rectify
{
 public:
  /// \cond
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \endcond

  /**
   * @brief Constructs an instance to compute the rectify maps from a
   * calibration file.
   *
   * @param calibration_filename Name of the calibration file.
   * @param alpha Free scaling parameter between 0 and 1.
   * See <a
   * href="https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1">cv::getOptimalNewCameraMatrix()</a>
   * for details.
   * @param width_sampling_factor Width sampling factor.
   * @param height_sampling_factor Height sampling factor.
   */
  Rectify(const std::string& calibration_filename, const float alpha,
          const float width_sampling_factor, const float height_sampling_factor)
  {
    ASSERT(0 <= alpha && alpha <= 1, "The free scaling parameter "
                                         << alpha
                                         << " must be between 0 and 1");
    ASSERT(1 <= width_sampling_factor, "The width sampling reduction factor "
                                           << width_sampling_factor
                                           << " must be greater or equal to 1");
    ASSERT(1 <= height_sampling_factor,
           "The height sampling reduction factor "
               << height_sampling_factor << " must be greater or equal to 1");

    // read calibration parameters
    uint16_t width, height;
    read_calibration(calibration_filename, width, height, original_intrinsics_,
                     distortion_parameters_);
    width_ = std::ceil(static_cast<float>(width) / width_sampling_factor);
    height_ = std::ceil(static_cast<float>(height) / height_sampling_factor);

    CvMatrix original_intrinsics;
    cv::eigen2cv(original_intrinsics_, original_intrinsics);
    original_intrinsics.row(0) /= width_sampling_factor;
    original_intrinsics.row(1) /= height_sampling_factor;
    CvMatrix distortion_parameters;
    cv::eigen2cv(distortion_parameters_, distortion_parameters);

    // compute new optimal intrinsics
    CvMatrix intrinsics = cv::getOptimalNewCameraMatrix(
        original_intrinsics, distortion_parameters, CvSize(width_, height_),
        alpha, CvSize(width_, height_));
    cv::cv2eigen(intrinsics, intrinsics_);

    // compute undistortion map
    CvMatrix map;
    cv::initInverseRectificationMap(
        original_intrinsics, distortion_parameters, CvMatrix(), intrinsics,
        CvSize(width_, height_), CV_32FC2, map_, map);
  }
  /**
   * @brief Deleted copy constructor.
   */
  Rectify(const Rectify&) = delete;
  /**
   * @brief Default move constructor.
   */
  Rectify(Rectify&&) = default;
  /**
   * @brief Deleted copy assignment operator.
   */
  Rectify&
  operator=(const Rectify&) = delete;
  /**
   * @brief Default move assignment operator.
   */
  Rectify&
  operator=(Rectify&&) = default;
  /**
   * @brief Default destructor.
   */
  ~Rectify() = default;

  /**
   * @brief Returns the width of the context.
   *
   * @return Width of the context.
   */
  uint16_t
  width() const
  {
    return width_;
  }

  /**
   * @brief Returns the height of the context.
   *
   * @return Height of the context.
   */
  uint16_t
  height() const
  {
    return height_;
  }

  /**
   * @brief Returns the undistorted camera intrinsics.
   *
   * @return Reference to the undistorted camera intrinsics.
   */
  const Matrix<float, 3, 3>&
  intrinsics() const
  {
    return intrinsics_;
  }

  /**
   * @brief Returns the original camera intrinsics.
   *
   * @return Reference to the original camera intrinsics.
   */
  const Matrix<float, 3, 3>&
  original_intrinsics() const
  {
    return original_intrinsics_;
  }

  /**
   * @brief Returns the camera distortion parameters.
   *
   * @return Reference to the camera distortion parameters.
   */
  const Vector<float, 5>&
  distortion_parameters() const
  {
    return distortion_parameters_;
  }

  /**
   * @brief Returns the undistorted coordinates.
   *
   * @param x x-coordinate.
   * @param y y-coordinate.
   * @return Reference to the undistorted coordinates.
   */
  const CvVector<float, 2>&
  map(const uint16_t x, const uint16_t y) const
  {
    ASSERT(x < width_,
           "The x coordinate " << x << " must be smaller than " << width_);
    ASSERT(y < height_,
           "The y coordinate " << y << " must be smaller than " << height_);
    return map_.at<CvVector<float, 2>>(y, x);
  }

 protected:
  /**
   * @brief Width of the context.
   */
  uint16_t width_;
  /**
   * @brief Height of the context.
   */
  uint16_t height_;

  /**
   * @brief Undistorted camera intrinsics.
   */
  Matrix<float, 3, 3> intrinsics_;
  /**
   * @brief Original camera intrinsics.
   */
  Matrix<float, 3, 3> original_intrinsics_;
  /**
   * @brief Camera distortion parameters.
   */
  Vector<float, 5> distortion_parameters_;

  /**
   * @brief Rectify maps.
   */
  CvMatrix map_;
};

/**
 * @brief Make function that creates an instance of ETTCM::Rectify.
 *
 * @param calibration_filename Name of the calibration file.
 * @param alpha Free scaling parameter between 0 and 1.
 * See <a
 * href="https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1">cv::getOptimalNewCameraMatrix()</a>
 * for details.
 * @param width_sampling_factor Width sampling factor.
 * @param height_sampling_factor Height sampling factor.
 *
 * @return Instance of ETTCM::Rectify.
 */
inline Rectify
make_rectify(const std::string& calibration_filename, const float alpha = 0,
             const float width_sampling_factor = 1,
             const float height_sampling_factor = 1)
{
  return Rectify(calibration_filename, alpha, width_sampling_factor,
                 height_sampling_factor);
}
}  // namespace ETTCM

#endif  // ETTCM_RECTIFY_HPP
