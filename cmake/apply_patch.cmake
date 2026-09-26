# Applies PATCH to the git checkout in the working directory unless it is
# already applied, so re-running the FetchContent patch step is harmless.
#   cmake -DPATCH=<file> -P apply_patch.cmake
execute_process(
  COMMAND git apply --reverse --check ${PATCH}
  RESULT_VARIABLE already_applied
  OUTPUT_QUIET ERROR_QUIET
)
if(NOT already_applied EQUAL 0)
  execute_process(COMMAND git apply ${PATCH} RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "failed to apply ${PATCH}")
  endif()
endif()
