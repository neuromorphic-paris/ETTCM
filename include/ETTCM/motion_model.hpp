/**
 * @file
 * @brief Motion models implementation.
 */

#ifndef ETTCM_MOTION_MODEL_HPP
#define ETTCM_MOTION_MODEL_HPP

#include <cstdint>

#include "ETTCM/types.hpp"
#include "ETTCM/utils.hpp"

namespace ETTCM
{
/**
 * @brief Scaling motion model.
 */
struct Scaling
{
  enum : uint8_t
  {
    NDims = 2, /**< Number of spatial dimensions. */
    NV = 1,    /**< Number of linear velocity parameters. */
    NVars = 1  /**< Number of motion model parameters. */
  };

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           [[maybe_unused]] const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << -p(0), -p(1)).finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           [[maybe_unused]] const Ref<const Vector<float, NDims>> f,
           const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << -inverse_depth * p(0),
            -inverse_depth * p(1))
        .finished();
  }

  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           [[maybe_unused]] const Ref<const Vector<float, NDims>> f)
  {
    return -t_diff * b_matrix(p, f);
  }
  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           [[maybe_unused]] const Ref<const Vector<float, NDims>> f,
           const float inverse_depth)
  {
    return -t_diff * b_matrix(p, f, inverse_depth);
  }
};

/**
 * @brief Driving motion model.
 */
struct Driving
{
  enum : uint8_t
  {
    NDims = 2, /**< Number of spatial dimensions. */
    NV = 1,    /**< Number of linear velocity parameters. */
    NVars = 2  /**< Number of motion model parameters. */
  };

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << -p(0), f(0) + p(0) * p(0) / f(0),
            -p(1), p(0) * p(1) / f(0))
        .finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << -inverse_depth * p(0),
            f(0) + p(0) * p(0) / f(0), -inverse_depth * p(1),
            p(0) * p(1) / f(0))
        .finished();
  }

  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return -t_diff * b_matrix(p, f);
  }
  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return -t_diff * b_matrix(p, f, inverse_depth);
  }
};

/**
 * @brief Translation in 3D motion model.
 */
struct Translation3D
{
  enum : uint8_t
  {
    NDims = 2, /**< Number of spatial dimensions. */
    NV = 3,    /**< Number of linear velocity parameters. */
    NVars = 3  /**< Number of motion model parameters. */
  };

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, without taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << f(0), 0, -p(0), 0, f(1), -p(1))
        .finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << inverse_depth * f(0), 0,
            -inverse_depth * p(0), 0, inverse_depth * f(1),
            -inverse_depth * p(1))
        .finished();
  }

  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, without taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return -t_diff * b_matrix(p, f);
  }
  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return -t_diff * b_matrix(p, f, inverse_depth);
  }
};

/**
 * @brief 6-DOF motion model.
 */
struct SixDOF
{
  enum : uint8_t
  {
    NDims = 2, /**< Number of spatial dimensions. */
    NV = 3,    /**< Number of linear velocity parameters. */
    NVars = 6  /**< Number of motion model parameters. */
  };

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, without taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << f(0), 0, -p(0),
            -p(0) * p(1) / f(1), f(0) + p(0) * p(0) / f(0), -p(1) * f(0) / f(1),
            0, f(1), -p(1), -(f(1) + p(1) * p(1) / f(1)), p(0) * p(1) / f(0),
            p(0) * f(1) / f(0))
        .finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << inverse_depth * f(0), 0,
            -inverse_depth * p(0), -p(0) * p(1) / f(1),
            f(0) + p(0) * p(0) / f(0), -p(1) * f(0) / f(1), 0,
            inverse_depth * f(1), -inverse_depth * p(1),
            -(f(1) + p(1) * p(1) / f(1)), p(0) * p(1) / f(0),
            p(0) * f(1) / f(0))
        .finished();
  }

  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, without taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return -t_diff * b_matrix(p, f);
  }
  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return -t_diff * b_matrix(p, f, inverse_depth);
  }
};

/**
 * @brief Parameterized 6-DOF motion model.
 */
struct ParameterizedSixDOF
{
  enum : uint8_t
  {
    NDims = 2, /**< Number of spatial dimensions. */
    NV = 3,    /**< Number of linear velocity parameters. */
    NVars = 6  /**< Number of motion model parameters. */
  };

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, without taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << f(0), 0, -p(0),
            -p(0) * p(1) / f(1), f(0) + p(0) * p(0) / f(0), -p(1) * f(0) / f(1),
            0, f(1), -p(1), -(f(1) + p(1) * p(1) / f(1)), p(0) * p(1) / f(0),
            p(0) * f(1) / f(0))
        .finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix(const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << inverse_depth * f(0), 0,
            -inverse_depth * p(0), -p(0) * p(1) / f(1),
            f(0) + p(0) * p(0) / f(0), -p(1) * f(0) / f(1), 0,
            inverse_depth * f(1), -inverse_depth * p(1),
            -(f(1) + p(1) * p(1) / f(1)), p(0) * p(1) / f(0),
            p(0) * f(1) / f(0))
        .finished();
  }

  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, without taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix_param(const Ref<const Vector<float, NDims>> p,
                 const Ref<const Vector<float, NDims>> f)
  {
    return (Matrix<float, NDims, NVars>() << f(0), 0, -p(0),
            -p(0) * p(1) / f(1), p(0) * p(0) / f(0), -p(1) * f(0) / f(1), 0,
            f(1), -p(1), -p(1) * p(1) / f(1), p(0) * p(1) / f(0),
            p(0) * f(1) / f(0))
        .finished();
  }
  /**
   * @brief Computes a specific matrix for the incremental motion model
   * parameters estimation, taking the inverse depth into account.
   *
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Specific matrix for the incremental motion model parameters
   * estimation.
   */
  static Matrix<float, NDims, NVars>
  b_matrix_param(const Ref<const Vector<float, NDims>> p,
                 const Ref<const Vector<float, NDims>> f,
                 const float inverse_depth)
  {
    return (Matrix<float, NDims, NVars>() << f(0), 0, -inverse_depth * p(0),
            -p(0) * p(1) / f(1), p(0) * p(0) / f(0), -p(1) * f(0) / f(1), 0,
            f(1), -inverse_depth * p(1), -p(1) * p(1) / f(1),
            p(0) * p(1) / f(0), p(0) * f(1) / f(0))
        .finished();
  }

  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, without taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f)
  {
    return -t_diff * (Matrix<float, NDims, NVars>() << f(0), 0, -p(0),
                      -p(0) * p(1) / f(1), f(0) + p(0) * p(0) / f(0),
                      -p(1) * f(0) / f(1), 0, f(1), -p(1),
                      -(f(1) + p(1) * p(1) / f(1)), p(0) * p(1) / f(0),
                      p(0) * f(1) / f(0))
                         .finished();
  }
  /**
   * @brief Computes the Jacobian for the incremental motion model parameters
   * estimation, taking the inverse depth into account.
   *
   * @param t_diff Time difference.
   * @param p Centered spatial coordinates.
   * @param f Focal parameters.
   * @param inverse_depth Inverse depth parameter.
   *
   * @return Jacobian for the incremental motion model parameters estimation.
   */
  static Matrix<float, NDims, NVars>
  jacobian(const float t_diff, const Ref<const Vector<float, NDims>> p,
           const Ref<const Vector<float, NDims>> f, const float inverse_depth)
  {
    return -t_diff * (Matrix<float, NDims, NVars>() << f(0), 0,
                      -inverse_depth * p(0), -p(0) * p(1) / f(1),
                      f(0) + p(0) * p(0) / f(0), -p(1) * f(0) / f(1), 0, f(1),
                      -inverse_depth * p(1), -(f(1) + p(1) * p(1) / f(1)),
                      p(0) * p(1) / f(0), p(0) * f(1) / f(0))
                         .finished();
  }
};
}  // namespace ETTCM

#endif  // ETTCM_MOTION_MODEL_HPP
