/**
 * @file
 * @brief Global (up-to scale) 3D motion and inverse depth estimator
 * implementation.
 */

#ifndef ETTCM_GLOBAL_MOTION_INVERSE_DEPTH_HPP
#define ETTCM_GLOBAL_MOTION_INVERSE_DEPTH_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ETTCM/assert.hpp"
#include "ETTCM/global_decay.hpp"
#include "ETTCM/motion_model.hpp"
#include "ETTCM/types.hpp"
#include "ETTCM/utils.hpp"

namespace ETTCM
{
/**
 * @brief Global motion parameters.
 */
struct GlobalMotionParams
{
  /**
   * @brief Theshold to stop the iterative estimation.
   */
  float d_min;
  /**
   * @brief Maximum number of iterations of the iterative estimation.
   */
  uint16_t number_iterations;
};

/**
 * @brief Timestamp and weight of an event.
 */
struct Cell
{
  /**
   * @brief Event timestamp \f$[\text{seconds}]\f$.
   */
  float t;
  /**
   * @brief Event weight.
   */
  float w;
};

/**
 * @brief Global (up-to scale) 3D motion and inverse depth estimator.
 *
 * @tparam Event Type of event.
 * @tparam MotionModel Type of global motion model to be estimated.
 * @tparam RectifyMap Type of the mapping for event rectification.
 * @tparam EventToGlobalMotion Type of the handle that passes from an event to
 * global motion parameters.
 * @tparam HandleGlobalMotion Type of the handle to further process the
 * estimated global motion.
 */
template <typename Event, typename MotionModel, typename RectifyMap,
          typename EventToGlobalMotion, typename HandleGlobalMotion>
class GlobalMotionInverseDepth
{
 public:
  /// \cond
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \endcond

  /**
   * @brief Type of the context, which consist on a surface whose elements
   * contain the most recent events.
   */
  typedef StdVector<Cell> Context;

  enum : uint8_t
  {
    NDims = MotionModel::NDims, /**< Number of spatial dimensions of the motion
                                   model. */
    NV = MotionModel::NV, /**< Number of parameters of the linear velocity.*/
    NVars =
        MotionModel::NVars, /**< Number of parameters of the motion model. */
    NModel = NVars + 1      /**< Number of parameters of the model. */
  };

  /**
   * @brief Constructs an instance to compute the global (up-to scale) 3D motion
   * parameters and inverse depth from an event stream.
   *
   * @param spatial_window @copybrief spatial_window_
   * @param weight_thresh @copybrief weight_thresh_
   * @param motion_vars_prior Motion uncertainty prior \f$\geq0\f$.
   * @param lambda_prior Parameterized inverse depth prior \f$\geq0\f$.
   * @param decay @copybrief decay_
   * @param params @copybrief params_
   * @param rectify_map @copybrief rectify_map_
   * @param event_to_global_motion @copybrief event_to_global_motion_
   * @param handle_global_motion @copybrief handle_global_motion_
   */
  GlobalMotionInverseDepth(const uint16_t spatial_window,
                           const float weight_thresh,
                           const float motion_vars_prior,
                           const float lambda_prior, const Decay& decay,
                           const GlobalMotionParams& params,
                           const RectifyMap& rectify_map,
                           EventToGlobalMotion&& event_to_global_motion,
                           HandleGlobalMotion&& handle_global_motion)
      : width_(rectify_map.width()),
        height_(rectify_map.height()),
        spatial_window_(spatial_window),
        window_size_((spatial_window_ << 1) + 1),
        weight_thresh_(weight_thresh),
        params_(params),
        motion_vars_(Vector<float, NModel>::Zero()),
        inverse_depth_(Matrix<float>::Ones(width_, height_)),
        context_on_(width_ * height_, Cell{-1, 0}),
        context_off_(width_ * height_, Cell{-1, 0}),
        decay_(decay),
        rectify_map_(rectify_map),
        event_to_global_motion_(
            std::forward<EventToGlobalMotion>(event_to_global_motion)),
        handle_global_motion_(
            std::forward<HandleGlobalMotion>(handle_global_motion))
  {
    ASSERT(0 <= weight_thresh_ && weight_thresh_ < 1,
           "The weight threshold " << weight_thresh_
                                   << " must be between 0 and 1");
    ASSERT(0 <= motion_vars_prior, "The motion prior "
                                       << motion_vars_prior
                                       << " must be greater than 0");
    ASSERT(0 <= lambda_prior,
           "The lambda prior " << lambda_prior << " must be greater than 0");

    prior_.template head<NVars>().array() = motion_vars_prior;
    prior_(NVars) = lambda_prior;
  }
  /**
   * @brief Deleted copy constructor.
   */
  GlobalMotionInverseDepth(const GlobalMotionInverseDepth&) = delete;
  /**
   * @brief Default move constructor.
   */
  GlobalMotionInverseDepth(GlobalMotionInverseDepth&&) = default;
  /**
   * @brief Deleted copy assignment operator.
   */
  GlobalMotionInverseDepth&
  operator=(const GlobalMotionInverseDepth&) = delete;
  /**
   * @brief Default move assignment operator.
   */
  GlobalMotionInverseDepth&
  operator=(GlobalMotionInverseDepth&&) = default;
  /**
   * @brief Default destructor.
   */
  ~GlobalMotionInverseDepth() = default;

  /**
   * @brief Incrementally computes the (up-to scale) translation in 3D motion
   * parameters one event at a time and respective inverse depth.
   *
   * This method incrementally estimates the (up-to scale) translation in 3D
   * motion parameters and inverse depth for each event in an event-by-event
   * basis.
   *
   * @param event Incoming event.
   */
  void
  operator()(Event event)
  {
    ASSERT(event.x_original < width_,
           "The event x coordinate " << event.x_original
                                     << " must be smaller than " << width_);
    ASSERT(event.y_original < height_,
           "The event y coordinate " << event.y_original
                                     << " must be smaller than " << height_);

    const Matrix<float, 3, 3>& intrinsics = rectify_map_.intrinsics();
    Context& context = (event.is_increase) ? context_on_ : context_off_;
    const float t = static_cast<float>(1e-6) * static_cast<float>(event.t);

    // compute specific incremental b matrix
    const Vector<float, NDims> p(event.x - intrinsics(0, 2),
                                 event.y - intrinsics(1, 2));
    Vector<float, NDims> v;
    float inverse_depth = 0;
    if constexpr (std::is_same<MotionModel, ParameterizedSixDOF>::value)
    {
      {
        const Matrix<float, NDims, NVars> b(MotionModel::b_matrix_param(
            p, intrinsics.diagonal().template head<NDims>()));
        v.noalias() = b * motion_vars_.template head<NVars>();
      }
      v.normalize();

      Vector<uint16_t, NDims> p_prev(event.x_original, event.y_original);
      p_prev -=
          (static_cast<float>(spatial_window_) * v).template cast<uint16_t>();

      // local window boundaries
      uint16_t x_min, x_max;
      uint16_t y_min, y_max;
      compute_local_window_boundaries(p_prev(0), p_prev(1), width_, height_,
                                      spatial_window_, spatial_window_, x_min,
                                      x_max, y_min, y_max);

      if (x_max >= x_min && y_max >= y_min)
      {
        float w_inverse_depth = 0;
        // update context weights
        for (uint16_t x = x_min; x <= x_max; ++x)
        {
          for (uint16_t y = y_min; y <= y_max; ++y)
          {
            Cell& context_cell = context[row_column_to_ind(x, y, height_)];
            if (context_cell.t >= 0)
            {
              const float t_diff = t - context_cell.t;
              context_cell.w =
                  static_cast<float>(1) /
                  (t_diff * decay_.n_decay + static_cast<float>(1));

              // if (context_cell.w >= weight_thresh_)
              {
                inverse_depth += context_cell.w * inverse_depth_(x, y);
                w_inverse_depth += context_cell.w;
              }
            }
          }
        }
        if (w_inverse_depth > 0)
        {
          inverse_depth /= w_inverse_depth;
        }
        else
        {
          inverse_depth = 1;
        }
        motion_vars_(NVars) = std::log(inverse_depth);

        // iterative estimation
        Vector<float, NV> v3(
            (motion_vars_(0) - motion_vars_(4)) / inverse_depth,
            (motion_vars_(1) + motion_vars_(3)) / inverse_depth,
            motion_vars_(2));
        const Vector<float, NModel> motion_vars(motion_vars_);
        Vector<float, NModel> num;
        Matrix<float, NModel, NModel> den;
        float d = params_.d_min;
        for (uint16_t k = 0;
             k < params_.number_iterations && d >= params_.d_min; ++k)
        {
          float w_sum = 0;
          num.setZero();
          den.setZero();

          for (uint16_t x = x_min; x <= x_max; ++x)
          {
            for (uint16_t y = y_min; y <= y_max; ++y)
            {
              const Cell& context_cell =
                  context[row_column_to_ind(x, y, height_)];
              const float t_diff = t - context_cell.t;

              if (t_diff > 0 && context_cell.w >= weight_thresh_)
              {
                const CvVector<float, 2>& p_map = rectify_map_.map(x, y);
                const Vector<float, NDims> p_centered(
                    p_map[0] - intrinsics(0, 2), p_map[1] - intrinsics(1, 2));

                Matrix<float, NDims, NModel> jac;
                jac.template leftCols<NVars>() = MotionModel::jacobian(
                    t_diff, p_centered,
                    intrinsics.diagonal().template head<NDims>(),
                    inverse_depth);
                {
                  const Matrix<float, NDims, NVars> b(MotionModel::b_matrix(
                      p_centered, intrinsics.diagonal().template head<NDims>(),
                      inverse_depth));
                  jac.col(NVars).noalias() = b.template leftCols<NV>() * v3;
                }

                const Vector<float, NDims> p_diff(event.x - p_map[0],
                                                  event.y - p_map[1]);
                const Matrix<float, NDims, NVars> b(MotionModel::b_matrix_param(
                    p_centered, intrinsics.diagonal().template head<NDims>(),
                    inverse_depth));
                const Vector<float, NDims> residual(
                    p_diff -
                    t_diff * (b * motion_vars_.template head<NVars>()));

                const float w =
                    context_cell.w *
                    compute_exp(-static_cast<float>(0.5) *
                                (residual.transpose() * residual)(0));
                w_sum += w;
                num.noalias() -= w * (jac.transpose() * residual);
                den.noalias() += (w * jac.transpose()) * jac;
              }
            }
          }

          if (w_sum > 0)
          {
            num.array() +=
                prior_.array() * (motion_vars - motion_vars_).array();
            den.diagonal() += prior_;
            const Vector<float, NModel> motion_vars_diff(den.llt().solve(num));
            motion_vars_ += motion_vars_diff;
            inverse_depth = std::exp(motion_vars_(NVars));
            v3 << (motion_vars_(0) - motion_vars_(4)) / inverse_depth,
                (motion_vars_(1) + motion_vars_(3)) / inverse_depth,
                motion_vars_(2);
            const float v_norm = v3.norm();
            if (v_norm > static_cast<float>(1))
            {
              v3 /= v_norm;
              motion_vars_(0) = inverse_depth * v3(0) + motion_vars_(4);
              motion_vars_(1) = inverse_depth * v3(1) - motion_vars_(3);
              motion_vars_(2) = v3(2);
            }
            d = motion_vars_diff.norm();
          }
          else
          {
            d = 0;
          }
        }

        inverse_depth_(event.x_original, event.y_original) = inverse_depth;
      }

      // compute local flow
      const Matrix<float, NDims, NVars> b(MotionModel::b_matrix_param(
          p, intrinsics.diagonal().template head<NDims>(), inverse_depth));
      v.noalias() = b * motion_vars_.template head<NVars>();
    }
    else
    {
      {
        const Matrix<float, NDims, NVars> b(MotionModel::b_matrix(
            p, intrinsics.diagonal().template head<NDims>()));
        v.noalias() = b * motion_vars_.template head<NVars>();
      }
      v.normalize();

      Vector<uint16_t, NDims> p_prev(event.x_original, event.y_original);
      p_prev -=
          (static_cast<float>(spatial_window_) * v).template cast<uint16_t>();

      // local window boundaries
      uint16_t x_min, x_max;
      uint16_t y_min, y_max;
      compute_local_window_boundaries(p_prev(0), p_prev(1), width_, height_,
                                      spatial_window_, spatial_window_, x_min,
                                      x_max, y_min, y_max);

      if (x_max >= x_min && y_max >= y_min)
      {
        float w_inverse_depth = 0;
        // update context weights
        for (uint16_t x = x_min; x <= x_max; ++x)
        {
          for (uint16_t y = y_min; y <= y_max; ++y)
          {
            Cell& context_cell = context[row_column_to_ind(x, y, height_)];
            if (context_cell.t >= 0)
            {
              const float t_diff = t - context_cell.t;
              context_cell.w =
                  static_cast<float>(1) /
                  (t_diff * decay_.n_decay + static_cast<float>(1));

              // if (context_cell.w >= weight_thresh_)
              {
                inverse_depth += context_cell.w * inverse_depth_(x, y);
                w_inverse_depth += context_cell.w;
              }
            }
          }
        }
        if (w_inverse_depth > 0)
        {
          inverse_depth /= w_inverse_depth;
        }
        else
        {
          inverse_depth = 1;
        }
        motion_vars_(NVars) = std::log(inverse_depth);

        // iterative estimation
        const Vector<float, NModel> motion_vars(motion_vars_);
        Vector<float, NModel> num;
        Matrix<float, NModel, NModel> den;
        float d = params_.d_min;
        for (uint16_t k = 0;
             k < params_.number_iterations && d >= params_.d_min; ++k)
        {
          float w_sum = 0;
          num.setZero();
          den.setZero();

          for (uint16_t x = x_min; x <= x_max; ++x)
          {
            for (uint16_t y = y_min; y <= y_max; ++y)
            {
              const Cell& context_cell =
                  context[row_column_to_ind(x, y, height_)];
              const float t_diff = t - context_cell.t;

              if (t_diff > 0 && context_cell.w >= weight_thresh_)
              {
                const CvVector<float, 2>& p_map = rectify_map_.map(x, y);
                const Vector<float, NDims> p_centered(
                    p_map[0] - intrinsics(0, 2), p_map[1] - intrinsics(1, 2));

                Matrix<float, NDims, NModel> jac;
                jac.template leftCols<NVars>() = MotionModel::jacobian(
                    t_diff, p_centered,
                    intrinsics.diagonal().template head<NDims>(),
                    inverse_depth);
                jac.col(NVars).noalias() = jac.template leftCols<NV>() *
                                           motion_vars_.template head<NV>();

                const Vector<float, NDims> p_diff(event.x - p_map[0],
                                                  event.y - p_map[1]);
                const Vector<float, NDims> residual(
                    p_diff + jac.template leftCols<NVars>() *
                                 motion_vars_.template head<NVars>());

                const float w =
                    context_cell.w *
                    compute_exp(-static_cast<float>(0.5) *
                                (residual.transpose() * residual)(0));
                w_sum += w;
                num.noalias() -= w * (jac.transpose() * residual);
                den.noalias() += (w * jac.transpose()) * jac;
              }
            }
          }

          if (w_sum > 0)
          {
            num.array() +=
                prior_.array() * (motion_vars - motion_vars_).array();
            den.diagonal() += prior_;
            const Vector<float, NModel> motion_vars_diff(den.llt().solve(num));
            motion_vars_ += motion_vars_diff;
            const float v_norm = motion_vars_.template head<NV>().norm();
            if constexpr (std::is_same<MotionModel, Driving>::value)
            {
              if (v_norm > static_cast<float>(0))
              {
                motion_vars_.template head<NV>() /= v_norm;
              }
            }
            else
            {
              if (v_norm > static_cast<float>(1))
              {
                motion_vars_.template head<NV>() /= v_norm;
              }
            }
            d = motion_vars_diff.norm();

            inverse_depth = std::exp(motion_vars_(NVars));
          }
          else
          {
            d = 0;
          }
        }

        inverse_depth_(event.x_original, event.y_original) = inverse_depth;
      }

      // compute local flow
      const Matrix<float, NDims, NVars> b(MotionModel::b_matrix(
          p, intrinsics.diagonal().template head<NDims>(), inverse_depth));
      v.noalias() = b * motion_vars_.template head<NVars>();
    }

    // add event to context
    context[row_column_to_ind(event.x_original, event.y_original, height_)] =
        Cell{t, 1};

    handle_global_motion_(event_to_global_motion_(
        event, motion_vars_.template head<NVars>(), v, inverse_depth));
  }

  /**
   * @brief Resets the context.
   */
  void
  reset()
  {
    motion_vars_.setZero();
    inverse_depth_.setOnes();
    context_on_.clear();
    context_on_.resize(width_ * height_, Cell{-1, 0});
    context_off_.clear();
    context_off_.resize(width_ * height_, Cell{-1, 0});
  }

 protected:
  /**
   * @brief Width of the context.
   */
  const uint16_t width_;
  /**
   * @brief Height of the context.
   */
  const uint16_t height_;
  /**
   * @brief Local window radius \f$[\text{pixel}]\f$.
   */
  const uint16_t spatial_window_;
  /**
   * @brief Local window size \f$[\text{pixel}]\f$.
   */
  const uint16_t window_size_;
  /**
   * @brief Weight threshold \f$[0,1)\f$.
   */
  const float weight_thresh_;

  /**
   * @brief Iterative estimation parameters.
   * \sa ETTCM::GlobalMotionParams
   */
  const GlobalMotionParams params_;

  /**
   * @brief Global motion variables estimate.
   */
  Vector<float, NModel> motion_vars_;

  /**
   * @brief Inverse depth estimates.
   */
  Matrix<float> inverse_depth_;

  /**
   * @brief Uncertainty prior.
   */
  Vector<float, NModel> prior_;

  /**
   * @brief Context with the most recent events with ON polarity.
   */
  Context context_on_;
  /**
   * @brief Context with the most recent events with OFF polarity.
   */
  Context context_off_;

  /**
   * @brief Reference to a variable storing a global decay estimate.
   * \sa ETTCM::Decay.
   */
  const Decay& decay_;

  /**
   * @brief Rectification map.
   * \sa ETTCM::Rectify.
   */
  const RectifyMap& rectify_map_;

  /**
   * @brief Handle that passes from an event to global motion parameters.
   */
  EventToGlobalMotion event_to_global_motion_;
  /**
   * @brief Handle to further process the estimated global motion
   * parameters.
   */
  HandleGlobalMotion handle_global_motion_;
};

/**
 * @brief Make function that creates an instance of
 * ETTCM::GlobalMotionInverseDepth.
 *
 * @tparam Event Type of event.
 * @tparam MotionModel Type of global motion model to be estimated.
 * @tparam RectifyMap Type of the mapping for event rectification.
 * @tparam EventToGlobalMotion Type of the handle that passes from an event to
 * global motion parameters.
 * @tparam HandleGlobalMotion Type of the handle to further process the
 * estimated global motion.
 *
 * @param spatial_window Local window radius \f$[\text{pixel}]\f$.
 * @param weight_thresh Weight threshold \f$[0,1)\f$.
 * @param motion_vars_prior Motion uncertainty prior \f$\geq0\f$.
 * @param lambda_prior Parameterized inverse depth uncertainty prior
 * \f$\geq0\f$.
 * @param decay Reference to a variable storing a global decay estimate.
 * \sa ETTCM::Decay.
 * @param params Iterative estimation parameters.
 * \sa ETTCM::GlobalMotionParams.
 * @param rectify_map Rectification map.
 * \sa ETTCM::Rectify.
 * @param event_to_global_motion Handle that passes from an event to global
 * motion parameters.
 * @param handle_global_motion Handle to further process the estimated global
 * motion parameters.
 *
 * @return Instance of ETTCM::GlobalMotionInverseDepth.
 */
template <typename Event, typename MotionModel, typename RectifyMap,
          typename EventToGlobalMotion, typename HandleGlobalMotion>
inline GlobalMotionInverseDepth<Event, MotionModel, RectifyMap,
                                EventToGlobalMotion, HandleGlobalMotion>
make_global_motion_inverse_depth(const uint16_t spatial_window,
                                 const float weight_thresh,
                                 const float motion_vars_prior,
                                 const float lambda_prior, const Decay& decay,
                                 const GlobalMotionParams& params,
                                 const RectifyMap& rectify_map,
                                 EventToGlobalMotion&& event_to_global_motion,
                                 HandleGlobalMotion&& handle_global_motion)
{
  return GlobalMotionInverseDepth<Event, MotionModel, RectifyMap,
                                  EventToGlobalMotion, HandleGlobalMotion>(
      spatial_window, weight_thresh, motion_vars_prior, lambda_prior, decay,
      params, rectify_map,
      std::forward<EventToGlobalMotion>(event_to_global_motion),
      std::forward<HandleGlobalMotion>(handle_global_motion));
}

/**
 * @brief Make function that creates an instance of
 * ETTCM::GlobalMotionInverseDepth.
 *
 * @tparam Event Type of event.
 * @tparam MotionModel Type of global motion model to be estimated.
 * @tparam RectifyMap Type of the mapping for event rectification.
 * @tparam EventToGlobalMotion Type of the handle that passes from an event to
 * global motion parameters.
 * @tparam HandleGlobalMotion Type of the handle to further process the
 * estimated global motion.
 *
 * @param spatial_window Local window radius \f$[\text{pixel}]\f$.
 * @param weight_thresh Weight threshold \f$[0,1)\f$.
 * @param motion_vars_prior Motion uncertainty prior \f$\geq0\f$.
 * @param lambda_prior Parameterized inverse depth uncertainty prior
 * \f$\geq0\f$.
 * @param decay Reference to a variable storing a global decay estimate.
 * \sa ETTCM::Decay.
 * @param rectify_map Rectification map.
 * \sa ETTCM::Rectify.
 * @param event_to_global_motion Handle that passes from an event to global
 * motion parameters.
 * @param handle_global_motion Handle to further process the estimated global
 * motion parameters.
 *
 * @return Instance of ETTCM::GlobalMotionInverseDepth.
 */
template <typename Event, typename MotionModel, typename RectifyMap,
          typename EventToGlobalMotion, typename HandleGlobalMotion>
inline GlobalMotionInverseDepth<Event, MotionModel, RectifyMap,
                                EventToGlobalMotion, HandleGlobalMotion>
make_global_motion_inverse_depth(const uint16_t spatial_window,
                                 const float weight_thresh,
                                 const float motion_vars_prior,
                                 const float lambda_prior, const Decay& decay,
                                 const RectifyMap& rectify_map,
                                 EventToGlobalMotion&& event_to_global_motion,
                                 HandleGlobalMotion&& handle_global_motion)
{
  return GlobalMotionInverseDepth<Event, MotionModel, RectifyMap,
                                  EventToGlobalMotion, HandleGlobalMotion>(
      spatial_window, weight_thresh, motion_vars_prior, lambda_prior, decay,
      {static_cast<float>(1.0e-4), 2}, rectify_map,
      std::forward<EventToGlobalMotion>(event_to_global_motion),
      std::forward<HandleGlobalMotion>(handle_global_motion));
}
}  // namespace ETTCM

#endif  // ETTCM_GLOBAL_MOTION_INVERSE_DEPTH_HPP
