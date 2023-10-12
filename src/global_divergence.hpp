#ifndef ETTCM_GLOBAL_DIVERGENCE_HPP
#define ETTCM_GLOBAL_DIVERGENCE_HPP

#include <string>

#include "ETTCM.hpp"
#include "pontella.hpp"
#include "sepia.hpp"

#define GLOBAL_DIVERGENCE(MotionModel)                                        \
  using namespace ETTCM;                                                      \
                                                                              \
  int main(int argc, char* argv[])                                            \
  {                                                                           \
    typedef sepia::dvs_event Event;                                           \
    typedef GlobalMotionInverseDepthEvent<MotionModel> GlobalMotionEvent;     \
    constexpr sepia::type Type = sepia::type::dvs;                            \
                                                                              \
    struct Arguments                                                          \
    {                                                                         \
      uint16_t spatial_window;                                                \
      uint64_t t_decay_first;                                                 \
      float weight_thresh;                                                    \
      float motion_vars_prior;                                                \
      float lambda_prior;                                                     \
      float width_sampling_factor;                                            \
      float height_sampling_factor;                                           \
    };                                                                        \
                                                                              \
    return pontella::main(                                                    \
        {"global_divergence is an accuracy benchmark for global motion and "  \
         "inverse depth global divergence estimation at provided timestamps", \
         "Usage: "                                                            \
         "./global_divergence [options] /path/to/input.es "                   \
         "/path/to/calibration.txt /path/to/divergence.txt "                  \
         "/path/to/timestamps.txt",                                           \
         "Available options:",                                                \
         "    -s s, --spatial-window s    sets the radius of the spatial "    \
         "window",                                                            \
         "                                    defaults to 3",                 \
         "    -t t, --time-decay-first t  sets the initial time decay",       \
         "                                    defaults to 10000",             \
         "    -e e, --weight-threshold e  sets the weight threshold",         \
         "                                    defaults to 0.01",              \
         "    -m m, --motion-prior m      sets the motion uncertainty prior", \
         "                                    defaults to 1000",              \
         "    -l l, --lambda-prior l      sets the parameterized inverse "    \
         "depth uncertainty prior",                                           \
         "                                    defaults to 1",                 \
         "    -w w, --width-sampling w    sets the width sampling factor",    \
         "                                    defaults to 1 (no sampling)",   \
         "    -g g, --height-sampling g   sets the height sampling factor",   \
         "                                    defaults to 1 (no sampling)",   \
         "    -h, --help                  shows this help message"},          \
        argc, argv, 4,                                                        \
        {{"spatial-window", {"s"}},                                           \
         {"time-decay-first", {"t"}},                                         \
         {"weight-threshold", {"e"}},                                         \
         {"motion-prior", {"m"}},                                             \
         {"lambda-prior", {"l"}},                                             \
         {"width-sampling", {"w"}},                                           \
         {"height-sampling", {"g"}}},                                         \
        {}, [&](pontella::command command) {                                  \
          const std::string& filename = command.arguments[0];                 \
          const auto header =                                                 \
              sepia::read_header(sepia::filename_to_ifstream(filename));      \
          const std::string& calibration_filename = command.arguments[1];     \
          const std::string& divergence_filename = command.arguments[2];      \
          const std::string& timestamps_filename = command.arguments[3];      \
                                                                              \
          Arguments arguments;                                                \
          arguments.spatial_window =                                          \
              extract_argument(command, "spatial-window", 3);                 \
          arguments.t_decay_first =                                           \
              extract_argument(command, "time-decay-first", 10000);           \
          arguments.weight_thresh =                                           \
              extract_argument(command, "weight-threshold", 0.01);            \
          arguments.motion_vars_prior =                                       \
              extract_argument(command, "motion-prior", 1000.0);              \
          arguments.lambda_prior =                                            \
              extract_argument(command, "lambda-prior", 1.0);                 \
          arguments.width_sampling_factor =                                   \
              extract_argument(command, "width-sampling", 1.0);               \
          arguments.height_sampling_factor =                                  \
              extract_argument(command, "height-sampling", 1.0);              \
                                                                              \
          auto distorted_to_undistorted = [](LeakyEvent event, float x,       \
                                             float y) -> RectifiedEvent {     \
            return {event.t, event.x, event.y, x, y, event.is_increase};      \
          };                                                                  \
                                                                              \
          auto rectify = make_rectify(calibration_filename, 0,                \
                                      arguments.width_sampling_factor,        \
                                      arguments.height_sampling_factor);      \
                                                                              \
          Decay event_decay;                                                  \
          auto handle_global_decay = [&](Decay decay) {                       \
            event_decay = decay;                                              \
          };                                                                  \
                                                                              \
          auto global_decay = make_global_decay<RectifiedEvent>(              \
              arguments.t_decay_first,                                        \
              [](RectifiedEvent event, float decay, float n_decay,            \
                 float t_decay, float rate) -> Decay {                        \
                return {event.t, decay, n_decay, t_decay, rate};              \
              },                                                              \
              handle_global_decay);                                           \
                                                                              \
          float ree = 0;                                                      \
          uint64_t number_measurements = 0;                                   \
          uint64_t t = 0;                                                     \
          float divergence_gt = 0;                                            \
          float divergence_num = 0;                                           \
          float divergence_den = 0;                                           \
          std::ifstream divergence_fin(                                       \
              filename_to_ifstream(divergence_filename));                     \
          std::ifstream timestamps_fin(                                       \
              filename_to_ifstream(timestamps_filename));                     \
          auto handle_global_motion = [&](GlobalMotionEvent motion) {         \
            if (motion.inverse_depth > 0)                                     \
            {                                                                 \
              divergence_num = event_decay.decay * divergence_num -           \
                               motion.motion_vars(MotionModel::NV - 1) *      \
                                   motion.inverse_depth;                      \
              divergence_den =                                                \
                  event_decay.decay * divergence_den + static_cast<float>(1); \
                                                                              \
              if (t == 0)                                                     \
              {                                                               \
                read_value<float>(divergence_fin, divergence_gt);             \
                read_value<uint64_t>(timestamps_fin, t);                      \
              }                                                               \
              else if (t <= motion.t)                                         \
              {                                                               \
                const float divergence = divergence_num / divergence_den;     \
                ree += std::abs(divergence - divergence_gt) / divergence_gt;  \
                ++number_measurements;                                        \
                                                                              \
                read_value<float>(divergence_fin, divergence_gt);             \
                read_value<uint64_t>(timestamps_fin, t);                      \
              }                                                               \
            }                                                                 \
          };                                                                  \
                                                                              \
          auto global_motion =                                                \
              make_global_motion_inverse_depth<RectifiedEvent, MotionModel>(  \
                  arguments.spatial_window, arguments.weight_thresh,          \
                  arguments.motion_vars_prior, arguments.lambda_prior,        \
                  event_decay, rectify,                                       \
                  [](RectifiedEvent event,                                    \
                     Vector<float, MotionModel::NVars> motion_vars,           \
                     Vector<float, MotionModel::NDims> v,                     \
                     float inverse_depth) -> GlobalMotionEvent {              \
                    return {event.t,                                          \
                            event.x_original,                                 \
                            event.y_original,                                 \
                            event.x,                                          \
                            event.y,                                          \
                            motion_vars,                                      \
                            v(0),                                             \
                            v(1),                                             \
                            inverse_depth,                                    \
                            event.is_increase};                               \
                  },                                                          \
                  handle_global_motion);                                      \
                                                                              \
          auto event_to_leaky_event = [](Event event, uint16_t x,             \
                                         uint16_t y) -> LeakyEvent {          \
            return {event.t, event.x, event.y, x, y, event.is_increase};      \
          };                                                                  \
                                                                              \
          auto leaky_sampling = make_leaky_sampling<Event>(                   \
              header.width, header.height, arguments.width_sampling_factor,   \
              arguments.height_sampling_factor, event_to_leaky_event,         \
              [&](LeakyEvent event) {                                         \
                const CvVector<float, 2>& p = rectify.map(event.x, event.y);  \
                                                                              \
                const short x = static_cast<short>(std::round(p[0]));         \
                const short y = static_cast<short>(std::round(p[1]));         \
                if (0 <= x && x < rectify.width() && 0 <= y &&                \
                    y < rectify.height())                                     \
                {                                                             \
                  const RectifiedEvent event_rectified(                       \
                      distorted_to_undistorted(event, p[0], p[1]));           \
                                                                              \
                  global_decay(event_rectified);                              \
                  global_motion(event_rectified);                             \
                }                                                             \
              });                                                             \
                                                                              \
          sepia::join_observable<Type>(sepia::filename_to_ifstream(filename), \
                                       leaky_sampling);                       \
                                                                              \
          ree *= static_cast<float>(100) / number_measurements;               \
          std::cout << "REE: " << ree << " (%)\n";                            \
        });                                                                   \
  }

#endif  // ETTCM_GLOBAL_DIVERGENCE_HPP
