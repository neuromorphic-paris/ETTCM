/**
 * @file
 * @brief Common types used by the library.
 */

#ifndef ETTCM_TYPES_HPP
#define ETTCM_TYPES_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ETTCM
{
/**
 * @brief Alias type for <a
 * href="https://en.cppreference.com/w/cpp/container/vector">std::vector</a>.
 */
template <typename T, typename Allocator = Eigen::aligned_allocator<T>>
using StdVector = typename std::vector<T, Allocator>;

/**
 * @brief Alias type for <a
 * href="https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html">cv::Mat</a>.
 */
typedef cv::Mat CvMatrix;
/**
 * @brief Alias type for <a
 * href="https://docs.opencv.org/4.x/d6/d50/classcv_1_1Size__.html">cv::Size</a>.
 */
typedef cv::Size CvSize;
/**
 * @brief Alias type for <a
 * href="https://docs.opencv.org/4.x/d6/dcf/classcv_1_1Vec.html">cv::Vec</a>.
 */
template <typename T, int N>
using CvVector = cv::Vec<T, N>;

/// \cond
#define CV_TYPE(T, N) CV_MAKETYPE(cv::DataDepth<T>::value, N)
/// \endcond

// Eigen definitions
/// \cond
constexpr int Dynamic = Eigen::Dynamic;
/// \endcond

/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html">Eigen::Array</a>.
 */
template <typename T, int Rows = Dynamic, int Cols = Dynamic>
using Array = typename Eigen::Array<T, Rows, Cols>;
/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html">Eigen::Matrix</a>.
 */
template <typename T, int Rows = Dynamic, int Cols = Dynamic>
using Matrix = typename Eigen::Matrix<T, Rows, Cols>;
/**
 * @brief Alias type representing a row vector.
 */
template <typename T, int Cols = Dynamic>
using RowVector = typename Eigen::Matrix<T, 1, Cols>;
/**
 * @brief Alias type representing a column vector.
 */
template <typename T, int Rows = Dynamic>
using Vector = typename Eigen::Matrix<T, Rows, 1>;

/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html">Eigen::DenseBase</a>.
 */
template <typename T>
using DenseBase = typename Eigen::DenseBase<T>;
/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1MatrixBase.html">Eigen::MatrixBase</a>.
 */
template <typename T>
using MatrixBase = typename Eigen::MatrixBase<T>;
/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1PlainObjectBase.html">Eigen::PlainObjectBase</a>.
 */
template <typename T>
using PlainObjectBase = typename Eigen::PlainObjectBase<T>;

/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html">Eigen::Map</a>.
 */
template <typename T>
using Map = typename Eigen::Map<T>;
/**
 * @brief Alias type for <a
 * href="https://eigen.tuxfamily.org/dox/classEigen_1_1Ref.html">Eigen::Ref</a>.
 */
template <typename T>
using Ref = typename Eigen::Ref<T>;

/**
 * @brief Event decay structure.
 */
struct Decay
{
  /**
   * @brief Previous timestamp \f$[\text{microseconds}]\f$.
   */
  uint64_t t;
  /**
   * @brief Event decay in \f$[0,1]\f$.
   */
  float decay;
  /**
   * @brief Auxiliary variable that counts the incoming number of events.
   */
  float n_decay;
  /**
   * @brief Auxiliary variable that estimates the event time decay
   * \f$[\text{microseconds}]\f$.
   */
  float t_decay;
  /**
   * @brief Estimated event rate \f$[\text{events}/\text{microseconds}]\f$.
   */
  float rate;

  /**
   * @brief Resets the context.
   *
   * @param t_decay_first Initial time rate assumption to bootstrap the rate
   * estimator \f$[\text{microseconds}]\f$.
   */
  void
  reset(const uint64_t t_decay_first)
  {
    t = 0;
    decay = 1;
    n_decay = 0;
    t_decay = t_decay_first;
    rate = 0;
  }
};

/**
 * @brief Structure representing a global motion and inverse depth event.
 *
 * @tparam MotionModel Type of global motion model.
 */
template <typename MotionModel>
struct GlobalMotionInverseDepthEvent
{
  /// \cond
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \endcond

  /**
   * @brief Timestamp of the event.
   */
  uint64_t t;
  /**
   * @brief Horizontal coordinate of the event before rectification.
   */
  uint16_t x_original;
  /**
   * @brief Vertical coordinate of the event before rectification.
   */
  uint16_t y_original;
  /**
   * @brief Horizontal coordinate of the event after rectification.
   */
  float x;
  /**
   * @brief Vertical coordinate of the event after rectification.
   */
  float y;
  /**
   * @brief Global motion parameters.
   */
  Vector<float, MotionModel::NVars> motion_vars;
  /**
   * @brief Local horizontal flow.
   */
  float vx;
  /**
   * @brief Local vertical flow.
   */
  float vy;
  /**
   * @brief Local inverse depth estimate.
   */
  float inverse_depth;
  /**
   * @brief Polarity of the event.
   */
  bool is_increase;
};

/**
 * @brief Structure representing a global motion, inverse depth and
 * time-to-contact event.
 *
 * @tparam MotionModel Type of global motion model.
 */
template <typename MotionModel>
struct GlobalMotionInverseDepthTimeToContactEvent
{
  /// \cond
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \endcond

  /**
   * @brief Timestamp of the event.
   */
  uint64_t t;
  /**
   * @brief Horizontal coordinate of the event before rectification.
   */
  uint16_t x_original;
  /**
   * @brief Vertical coordinate of the event before rectification.
   */
  uint16_t y_original;
  /**
   * @brief Horizontal coordinate of the event after rectification.
   */
  float x;
  /**
   * @brief Vertical coordinate of the event after rectification.
   */
  float y;
  /**
   * @brief Global motion parameters.
   */
  Vector<float, MotionModel::NVars> motion_vars;
  /**
   * @brief Time-to-contact parameter.
   */
  float time_to_contact;
  /**
   * @brief Local horizontal flow.
   */
  float vx;
  /**
   * @brief Local vertical flow.
   */
  float vy;
  /**
   * @brief Local inverse depth estimate.
   */
  float inverse_depth;
  /**
   * @brief Polarity of the event.
   */
  bool is_increase;
};

/**
 * @brief Structure representing a leaky event.
 */
struct __attribute__((__packed__)) LeakyEvent
{
  /**
   * @brief Timestamp of the event.
   */
  uint64_t t;
  /**
   * @brief Horizontal coordinate of the event before leakage.
   */
  uint16_t x_original;
  /**
   * @brief Vertical coordinate of the event before leakage.
   */
  uint16_t y_original;
  /**
   * @brief Horizontal coordinate of the event after leakage.
   */
  uint16_t x;
  /**
   * @brief Vertical coordinate of the event after leakage.
   */
  uint16_t y;
  /**
   * @brief Polarity of the event.
   */
  bool is_increase;
};

/**
 * @brief Structure representing an event rectified.
 */
struct __attribute__((__packed__)) RectifiedEvent
{
  /**
   * @brief Timestamp of the event.
   */
  uint64_t t;
  /**
   * @brief Horizontal coordinate of the event before rectification.
   */
  uint16_t x_original;
  /**
   * @brief Vertical coordinate of the event before rectification.
   */
  uint16_t y_original;
  /**
   * @brief Horizontal coordinate of the event after rectification.
   */
  float x;
  /**
   * @brief Vertical coordinate of the event after rectification.
   */
  float y;
  /**
   * @brief Polarity of the event.
   */
  bool is_increase;
};
}  // namespace ETTCM

#endif  // ETTCM_TYPES_HPP
