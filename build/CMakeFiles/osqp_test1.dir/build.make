# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lxl/project/matrixlib_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lxl/project/matrixlib_test/build

# Include any dependencies generated for this target.
include CMakeFiles/osqp_test1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/osqp_test1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/osqp_test1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/osqp_test1.dir/flags.make

CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o: CMakeFiles/osqp_test1.dir/flags.make
CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o: ../src/osqp/osqp_test1.cpp
CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o: CMakeFiles/osqp_test1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lxl/project/matrixlib_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o -MF CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o.d -o CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o -c /home/lxl/project/matrixlib_test/src/osqp/osqp_test1.cpp

CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lxl/project/matrixlib_test/src/osqp/osqp_test1.cpp > CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.i

CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lxl/project/matrixlib_test/src/osqp/osqp_test1.cpp -o CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.s

# Object files for target osqp_test1
osqp_test1_OBJECTS = \
"CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o"

# External object files for target osqp_test1
osqp_test1_EXTERNAL_OBJECTS =

osqp_test1: CMakeFiles/osqp_test1.dir/src/osqp/osqp_test1.cpp.o
osqp_test1: CMakeFiles/osqp_test1.dir/build.make
osqp_test1: /usr/local/lib/libOsqpEigen.so.0.8.1
osqp_test1: /usr/local/lib/libosqp.so
osqp_test1: CMakeFiles/osqp_test1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lxl/project/matrixlib_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable osqp_test1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/osqp_test1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/osqp_test1.dir/build: osqp_test1
.PHONY : CMakeFiles/osqp_test1.dir/build

CMakeFiles/osqp_test1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/osqp_test1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/osqp_test1.dir/clean

CMakeFiles/osqp_test1.dir/depend:
	cd /home/lxl/project/matrixlib_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lxl/project/matrixlib_test /home/lxl/project/matrixlib_test /home/lxl/project/matrixlib_test/build /home/lxl/project/matrixlib_test/build /home/lxl/project/matrixlib_test/build/CMakeFiles/osqp_test1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/osqp_test1.dir/depend
