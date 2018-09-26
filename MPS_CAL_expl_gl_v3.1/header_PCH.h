#ifndef HEADER_PCH_H_INCLUDED
#define HEADER_PCH_H_INCLUDED

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#define GLEW_STATIC
#include <GL/glew.h>

#ifdef _WIN32
#include <GL/wglew.h> // For wglSwapInterval
#endif

#define FREEGLUT_STATIC
#include <GL/freeglut.h>
//#include <GL/glut.h>

#define GLM_FORCE_RADIANS

#include "../common/glm/glm.hpp"
#include "../common/glm/gtc/type_ptr.hpp"
#include "../common/glm/gtx/transform.hpp"
#include "../common/glm/gtx/quaternion.hpp"

#endif // HEADER_PCH_H_INCLUDED
