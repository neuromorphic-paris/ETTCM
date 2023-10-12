/**
 * @file
 * @brief Custom assert message for debug purposes.
 */

#ifndef ETTCM_ASSERT_HPP
#define ETTCM_ASSERT_HPP

#include <exception>
#include <iostream>

namespace ETTCM
{
/// \cond
#define NO_OP

#ifndef NDEBUG
#define ASSERT(condition, message)                                   \
  if (!(condition))                                                  \
  {                                                                  \
    std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
              << " line " << __LINE__ << ": " << message << '\n';    \
    std::terminate();                                                \
  }
#else
#define ASSERT(condition, message) NO_OP
#endif
/// \endcond
}  // namespace ETTCM

#endif  // ETTCM_ASSERT_HPP
