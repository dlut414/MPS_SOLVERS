/*
LICENCE
*/
//Renderer.cpp
//implementation of class Renderer

#include "Renderer.h"
using namespace mytype;

GLuint      Renderer::programID1;
GLuint      Renderer::programID2;
GLuint      Renderer::matrixID[5];
GLuint      Renderer::scalaID[4];
GLuint      Renderer::vertexArrayID;
GLuint      Renderer::vertexBufferPos;
GLuint      Renderer::vertexBufferType;
GLuint      Renderer::vertexBufferScreen;
/*-----to be improved-----*/
GLuint      Renderer::vertexBufferTriangle;
GLuint      Renderer::vertexBufferNorm;
GLuint      Renderer::vertexBufferAlpha;
/*------------------------*/

GLuint      Renderer::frameBuffer;
GLuint      Renderer::depthTex;
GLuint      Renderer::colorTex;
GLuint      Renderer::depth;

real3*      Renderer::r3_vertex_buffer_data;
integer*    Renderer::u_vertex_buffer_data;
/*-----to be improved-----*/
real3**     Renderer::r3_triangle_dev_data;
real3**     Renderer::r3_norm_dev_data;
real**      Renderer::r_alpha_dev_data;
real3*      Renderer::r3_triangle_host_data;
real3*      Renderer::r3_norm_host_data;
real*       Renderer::r_alpha_host_data;
/*------------------------*/

struct cudaGraphicsResource* Renderer::cuda_triangleVBO_resource;
struct cudaGraphicsResource* Renderer::cuda_normVBO_resource;
struct cudaGraphicsResource* Renderer::cuda_alphaVBO_resource;

real        Renderer::r_screen_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		-1.0f,  1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f
};

Camera      Renderer::g_Camera;
glm::ivec2  Renderer::g_MousePos;
glm::quat   Renderer::g_Rotation   = glm::angleAxis<float>( -PI * 0.5f, glm::vec3(1, 0, 0) );
glm::vec3   Renderer::g_Scale      = glm::vec3(2.5f);
glm::vec3   Renderer::g_Pan        = glm::vec3(-0.8f, -0.8f, 0.0f);
glm::vec3   Renderer::g_InitialCameraPosition = glm::vec3(0.0f, 0.0f, 2.5f);
glm::quat   Renderer::g_InitialCameraRotation;
glm::mat4   Renderer::g_mvp;
glm::mat4   Renderer::g_modelMat;
glm::mat4   Renderer::g_viewMat;
glm::mat4   Renderer::g_projectionMat;
glm::mat4   Renderer::g_viewModelMat;
glm::mat4   Renderer::g_projectionMatInv;
GLfloat     Renderer::f_pointRadius     = 5.0f;
GLfloat     Renderer::f_pointScale      = 5.0f;
GLfloat     Renderer::f_near            = 0.1f;
GLfloat     Renderer::f_far             = 10.0f;

MPS_GPU     Renderer::mps;
bool        Renderer::b_stop            = true;
bool        Renderer::b_leave           = false;
bool        Renderer::b_point           = false;
float       Renderer::f_width           = 1024;
float       Renderer::f_height          = 1024;
float       Renderer::f_scaleVel        = 0.005f;
float       Renderer::f_panVel          = 0.001f;
int         Renderer::i_mouseButton     = GLUT_LEFT_BUTTON;

e_key myKey;

float       Renderer::f_timer   = 0.0f;
int         Renderer::i_count   = 0;
int         Renderer::i_limit   = 100;
cudaEvent_t Renderer::cuda_start;
cudaEvent_t Renderer::cuda_stop;

inline cudaError_t checkCuda(cudaError_t result, const char* fun)
{
#ifdef DEBUG
    if(result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s -> in %s \n",
                cudaGetErrorString(result), fun);
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

void Renderer::myInit(int argc, char** argv)
{
    glutInit                (&argc, argv);
    glutInitDisplayMode     (GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition  (0, 0);
    glutInitWindowSize      (f_width, f_height);
    glutCreateWindow        ("mps");
    glLightModeli           (GL_LIGHT_MODEL_TWO_SIDE, 1);
    glutMouseFunc           (myMouse);
    glutMotionFunc          (myMotion);
    //glutMouseWheelFunc      (myMouseWheel);
    glutReshapeFunc         (myReshape);
    glutKeyboardFunc        (myKeyboard);
    glutDisplayFunc         (myDisplay);

    glEnable                (GL_TEXTURE_2D);
    glEnable                (GL_CULL_FACE);
    //glDisable               (GL_CULL_FACE);
    glFrontFace             (GL_CW);
    //glEnable                (GL_POINT_SPRITE_ARB);
    //glEnable                (GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable                (GL_DEPTH_TEST);
    glDepthFunc             (GL_LESS);
    glEnable                (GL_ALPHA_TEST);
    glAlphaFunc             (GL_GREATER, 0.01f);
    //glEnable                (GL_BLEND);
    //glBlendFunc             (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor            (1.0f, 1.0f, 1.0f, 0.0f);

    mps.Initial             ();

    glewInit                ();
    myInitVAO               ();
    myInitVBO               ();
    //myInitFramebuffer       ();
    //myInitTexture           ();

    programID1 = myLoadShader("./shaders1/vertex1.glsl", "./shaders1/fragment1.glsl");
    programID2 = myLoadShader("./shaders2/vertex2.glsl", "./shaders2/fragment2.glsl");

    matrixID[0] = glGetUniformLocation (programID1, "projectionMat");
    matrixID[1] = glGetUniformLocation (programID1, "viewModelMat");
    scalaID[0]  = glGetUniformLocation (programID1, "pointRadius");
    scalaID[1]  = glGetUniformLocation (programID1, "pointScale");
    //scalaID[2]  = glGetUniformLocation (programID_depth, "near");
    //scalaID[3]  = glGetUniformLocation (programID_depth, "far");

    //programID_normal = myLoadShader("./shaders/normal_vertex.glsl", "./shaders/normal_fragment.glsl");

    //matrixID[2] = glGetUniformLocation (programID_normal, "projectionMatInv");

    g_Camera.SetPosition        (g_InitialCameraPosition);
    g_Camera.SetProjectionRH    (45.0f, 1.0f, f_near, f_far);

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

}

void Renderer::myFinal()
{
    //glDeleteBuffers(1, &vertexBufferPos);
    //glDeleteBuffers(1, &vertexBufferType);
    /*-----to be improved-----*/
    /*
    glDeleteBuffers(1, &vertexBufferTriangle);
    glDeleteBuffers(1, &vertexBufferNorm);
    glDeleteBuffers(1, &vertexBufferAlpha);
    */
    /*------------------------*/
    /*-----improved-----*/
    myDeleteVBO(&vertexBufferTriangle, &cuda_triangleVBO_resource);
    myDeleteVBO(&vertexBufferNorm, &cuda_normVBO_resource);
    myDeleteVBO(&vertexBufferAlpha, &cuda_alphaVBO_resource);
    /*------------------*/
    //glDeleteTextures(1, &depthTex);
    //glDeleteTextures(1, &colorTex);

}

void Renderer::myInitVAO()
{
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
}

void Renderer::myInitVBO()
{
    /*
    r3_triangle_host_data = mps.vox_mc.getTriangle();
    r3_norm_host_data     = mps.vox_mc.getNorm();
    r_alpha_host_data     = mps.vox_mc.getAlpha();
    */
    /*-----improved-----*/
    r3_triangle_dev_data  = mps.getDevTriangle();
    r3_norm_dev_data      = mps.getDevNorm();
    r_alpha_dev_data      = mps.getDevAlpha();

    r3_triangle_host_data = mps.vox_mc.getTriangle();
    r3_norm_host_data     = mps.vox_mc.getNorm();
    r_alpha_host_data     = mps.vox_mc.getAlpha();
    /*------------------*/

    //glGenBuffers(1, &vertexBufferPos);
    //glGenBuffers(1, &vertexBufferType);

    /*-----to be improved-----*/
    /*
    glGenBuffers(1, &vertexBufferTriangle);
    glGenBuffers(1, &vertexBufferNorm);
    glGenBuffers(1, &vertexBufferAlpha);
    */
    /*------------------------*/
    /*-----improved-----*/
    myCreateVBO(&vertexBufferTriangle, mps.vox_mc.getMaxEdge()*sizeof(real3));
    checkCuda(cudaGraphicsGLRegisterBuffer(&cuda_triangleVBO_resource, vertexBufferTriangle, cudaGraphicsRegisterFlagsNone), "214");

    myCreateVBO(&vertexBufferNorm, mps.vox_mc.getMaxEdge()*sizeof(real3));
    checkCuda(cudaGraphicsGLRegisterBuffer(&cuda_normVBO_resource, vertexBufferNorm, cudaGraphicsRegisterFlagsNone), "217");

    myCreateVBO(&vertexBufferAlpha, mps.vox_mc.getMaxEdge()*sizeof(real));
    checkCuda(cudaGraphicsGLRegisterBuffer(&cuda_alphaVBO_resource, vertexBufferAlpha, cudaGraphicsRegisterFlagsNone), "220");
    /*------------------*/

    //glGenBuffers(1, &vertexBufferScreen);
    //glBindBuffer(GL_ARRAY_BUFFER, vertexBufferScreen);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(r_screen_vertex_buffer_data), r_screen_vertex_buffer_data, GL_STATIC_DRAW);
}

void Renderer::myInitFramebuffer()
{
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
}

void Renderer::myInitTexture()
{
    ///color texture
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, f_width, f_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTex, 0);

    ///depth texture
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, f_width, f_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, depthTex, 0);

    ///depth texture
    glGenTextures(1, &depth);
    glBindTexture(GL_TEXTURE_2D, depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, f_width, f_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth, 0);

    ///list of draw buffers
    GLenum drawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, drawBuffers);
}

void Renderer::myCreateVBO(GLuint* vbo, unsigned size)
{
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutReportErrors();
}

void Renderer::myDeleteVBO(GLuint* vbo, struct cudaGraphicsResource** cuda_resource)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    *vbo = 0;

    cudaGraphicsUnregisterResource(*cuda_resource);
}

void Renderer::myDrawBuffer_screen()
{
    glBindFramebuffer           (GL_FRAMEBUFFER, frameBuffer);

    ///use program
    glClear                     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram                (programID1);

    ///send data
    glUniformMatrix4fv          (matrixID[0], 1, GL_FALSE, &g_projectionMat[0][0]);
    glUniformMatrix4fv          (matrixID[1], 1, GL_FALSE, &g_viewModelMat[0][0]);
    //glUniform1f                 (scalaID[0], f_pointRadius);
    //glUniform1f                 (scalaID[1], f_pointScale);
    //glUniform1f                 (scalaID[2], f_near);
    //glUniform1f                 (scalaID[3], f_far);

    /*
    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferPos);
    glBufferData                (GL_ARRAY_BUFFER, mps.getNp()*sizeof(real3), r3_vertex_host_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferType);
    glBufferData                (GL_ARRAY_BUFFER, mps.getNp()*sizeof(real3), u_vertex_host_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (1, 1, GL_UNSIGNED_INT, GL_FALSE, 0, (void*)0);
    */
    /*-----to be improved-----*/
    /*
    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferTriangle);
    glBufferData                (GL_ARRAY_BUFFER, mps.vox_mc.getMaxEdge()*sizeof(real3), r3_triangle_host_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferNorm);
    glBufferData                (GL_ARRAY_BUFFER, mps.vox_mc.getMaxEdge()*sizeof(real3), r3_norm_host_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferAlpha);
    glBufferData                (GL_ARRAY_BUFFER, mps.vox_mc.getMaxEdge()*sizeof(real), r_alpha_host_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
    */
    /*------------------------*/

    ///draw
    /*-----to be improved-----*/
    /*
    glEnableVertexAttribArray   (0);
    glEnableVertexAttribArray   (1);
    glEnableVertexAttribArray   (2);

    //glDrawArrays                (GL_POINTS, 0, mps.getNp());
    glDrawArrays                (GL_TRIANGLES, 0, mps.vox_mc.getMaxEdge());

    glDisableVertexAttribArray  (0);
    glDisableVertexAttribArray  (1);
    glDisableVertexAttribArray  (2);
    */
    /*------------------------*/

    /*-----improved-----*/
    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferTriangle);
    glVertexAttribPointer       (0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferNorm);
    glVertexAttribPointer       (1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferAlpha);
    glVertexAttribPointer       (2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glEnableVertexAttribArray   (0);
    glEnableVertexAttribArray   (1);
    glEnableVertexAttribArray   (2);

    glDrawArrays                (GL_TRIANGLES, 0, mps.vox_mc.getMaxEdge());

    glDisableVertexAttribArray  (0);
    glDisableVertexAttribArray  (1);
    glDisableVertexAttribArray  (2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    /*------------------*/
}

void Renderer::myReshape(int width, int height)
{
    glViewport      (0, 0, width, width);
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();
    gluPerspective  (-90.0f, f_width/f_height, 1.0f, 100.0f);
}

void Renderer::myKeyboard(unsigned char key,int,int)
{
    glutPostRedisplay();

    switch(key)
    {
        //esc
        case 0x1b:
        {
            b_leave = true; break;
        }

        //enter
        case 0x0d:
        {
            b_stop = !b_stop;
            if(b_stop)
            {
                mps.WriteCase();
            }
            break;
        }

        //p
        case 0x70:
        {
            b_point = !b_point; break;
        }

        //space
        case 0x20:
        {
            g_Rotation = glm::angleAxis<float>( -PI * 0.5f, glm::vec3(1, 0, 0) );
            g_Scale = glm::vec3(2.5f);
            g_Pan = glm::vec3(-0.8f, -0.8f, 0.0f);
            break;
        }
    }
}

void Renderer::myMouse(int button, int state, int x, int y)
{
    g_MousePos = glm::ivec2(x, y);

    switch(button)
    {
        case GLUT_LEFT_BUTTON:
        {
            i_mouseButton = GLUT_LEFT_BUTTON;
            break;
        }
        case GLUT_RIGHT_BUTTON:
        {
            i_mouseButton = GLUT_RIGHT_BUTTON;
            break;
        }
        case GLUT_MIDDLE_BUTTON:
        {
            i_mouseButton = GLUT_MIDDLE_BUTTON;
            break;
        }
    }

}

void Renderer::myMotion(int x, int y)
{
    glm::ivec2 mousePos     = glm::ivec2( x, y );
    glm::vec2 delta         = glm::vec2( mousePos - g_MousePos );
    g_MousePos = mousePos;

    switch(i_mouseButton)
    {
        case GLUT_LEFT_BUTTON:
        {
            glm::quat rotX  = glm::angleAxis<float>( glm::radians(delta.y) * 0.5f, glm::vec3(1, 0, 0) );
            glm::quat rotY  = glm::angleAxis<float>( glm::radians(delta.x) * 0.5f, glm::vec3(0, 1, 0) );
            g_Rotation      = ( rotX * rotY ) * g_Rotation;
            break;
        }
        case GLUT_RIGHT_BUTTON:
        {
            g_Pan       += glm::vec3(f_panVel*delta.x, -f_panVel*delta.y, 0.0f);
            break;
        }
        case GLUT_MIDDLE_BUTTON:
        {
            g_Scale     += glm::vec3(delta.y * f_scaleVel);
            break;
        }
    }
}

void Renderer::myMouseWheel(int button, int dir, int x, int y)
{
    g_Scale *= dir * f_scaleVel;
}

void Renderer::myDraw()
{
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();

    glm::mat4 modelMatrix = glm::translate  (glm::mat4(1.0f), g_Pan )
                          * glm::toMat4     (g_Rotation)
                          * glm::scale      (glm::mat4(1.0f), g_Scale );

    g_viewModelMat      = g_Camera.GetViewMatrix() * modelMatrix;
    g_projectionMat     = g_Camera.GetProjectionMatrix();
    g_projectionMatInv  = glm::inverse(g_projectionMat);

    myDrawBuffer_screen ();

    glutSwapBuffers     ();

    glutReportErrors    ();

    myFps();
}

void Renderer::myDisplay()
{
    if(!b_stop)
    {
        size_t num_bytes;

        checkCuda(cudaGraphicsMapResources(1, &cuda_triangleVBO_resource, 0), "527");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)r3_triangle_dev_data, &num_bytes, cuda_triangleVBO_resource), "528");

        checkCuda(cudaGraphicsMapResources(1, &cuda_normVBO_resource, 0), "530");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)r3_norm_dev_data, &num_bytes, cuda_normVBO_resource), "531");

        checkCuda(cudaGraphicsMapResources(1, &cuda_alphaVBO_resource, 0), "533");
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)r_alpha_dev_data, &num_bytes, cuda_alphaVBO_resource), "534");

        mps.step();

        checkCuda(cudaGraphicsUnmapResources(1, &cuda_triangleVBO_resource, 0), "538");
        checkCuda(cudaGraphicsUnmapResources(1, &cuda_normVBO_resource, 0), "539");
        checkCuda(cudaGraphicsUnmapResources(1, &cuda_alphaVBO_resource, 0), "540");
    }

    myDraw();

    glutPostRedisplay();

    if(b_leave)
    {
        glutLeaveMainLoop();
    }
}

void Renderer::myMainLoop()
{
    glutMainLoop    ();
}

void Renderer::setLightPosition()
{
    int pos[] = {0.0f, 0.0f, 1.0f, 0.0f};

    glLightiv(GL_LIGHT0, GL_POSITION, pos);
}

void Renderer::myFps()
{
    i_count++;

    if(i_count == i_limit)
    {
        char    _str[256];
        float   _fps;

        cudaEventRecord(cuda_stop);
        cudaEventSynchronize(cuda_stop);
        cudaEventElapsedTime(&_fps, cuda_start, cuda_stop);

        _fps        = i_limit * 1000.0f / _fps;

        sprintf(_str, "dambreak ->     particles: %d    interp points: %d   voxel: %d   fps: %4.1f", mps.getNp(), mps.vox_mc.getNumVertex(), mps.vox_mc.getNumVoxel(), _fps);
        glutSetWindowTitle(_str);

        i_count     = 0;
        cudaEventRecord(cuda_start);
    }
}

GLuint Renderer::myLoadShader(const char* vertex_file_path, const char* fragment_file_path)
{
    GLuint VertexShaderID       = glCreateShader(GL_VERTEX_SHADER);
    GLuint FragmentShaderID     = glCreateShader(GL_FRAGMENT_SHADER);

	std::string     VertexShaderCode;
	std::ifstream   VertexShaderStream(vertex_file_path, std::ios::in);
	if(VertexShaderStream.is_open())
	{
		std::string Line = "";
		while(getline(VertexShaderStream, Line))
		{
			VertexShaderCode += "\n" + Line;
        }
		VertexShaderStream.close();
	}
	else
	{
		printf("Impossible to open %s. \n", vertex_file_path);
		getchar();
		return 0;
	}

	std::string     FragmentShaderCode;
	std::ifstream   FragmentShaderStream(fragment_file_path, std::ios::in);
	if(FragmentShaderStream.is_open())
	{
		std::string Line = "";
		while(getline(FragmentShaderStream, Line))
		{
			FragmentShaderCode += "\n" + Line;
        }
		FragmentShaderStream.close();
	}
    else
	{
		printf("Impossible to open %s. \n", fragment_file_path);
		getchar();
		return 0;
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource  (VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader (VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv   (VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv   (VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 )
	{
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource  (FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader (FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv   (FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv   (FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 )
	{
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader  (ProgramID, VertexShaderID);
	glAttachShader  (ProgramID, FragmentShaderID);
	glLinkProgram   (ProgramID);

	// Check the program
	glGetProgramiv  (ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv  (ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 )
	{
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	glDeleteShader  (VertexShaderID);
	glDeleteShader  (FragmentShaderID);

	return ProgramID;
}






