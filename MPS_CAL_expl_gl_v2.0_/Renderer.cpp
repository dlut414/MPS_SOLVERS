/*
LICENCE
*/
//Renderer.cpp
//implementation of class Renderer
#include "Renderer.h"
using namespace mytype;

GLuint      Renderer::programID_depth;
GLuint      Renderer::programID_normal;
GLuint      Renderer::matrixID[5];
GLuint      Renderer::scalaID[4];
GLuint      Renderer::vertexArrayID;
GLuint      Renderer::vertexBufferPos;
GLuint      Renderer::vertexBufferType;
GLuint      Renderer::vertexBufferScreen;

GLuint      Renderer::frameBuffer;
GLuint      Renderer::depthTex;
GLuint      Renderer::colorTex;
GLuint      Renderer::depth;

real3*      Renderer::r3_vertex_buffer_data;
integer*    Renderer::u_vertex_buffer_data;
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
glm::vec3   Renderer::g_Scale      = glm::vec3(1.8f);
glm::vec3   Renderer::g_Pan        = glm::vec3(-0.6f, -0.8f, 0.0f);
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
bool        Renderer::stop              = true;
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
    glutDisplayFunc         (myAct);

    glEnable                (GL_TEXTURE_2D);
    glEnable                (GL_CULL_FACE);
    glEnable                (GL_POINT_SPRITE_ARB);
    glEnable                (GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable                (GL_DEPTH_TEST);
    glEnable                (GL_ALPHA_TEST);
    glAlphaFunc             (GL_GREATER, 0.01f);
    glEnable                (GL_BLEND);
    //glBlendFunc             (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor            (1.0f, 1.0f, 1.0f, 0.0f);

    mps.Initial             ();

    glewInit                ();
    myInitVAO               ();
    myInitVBO               ();
    myInitFramebuffer       ();
    myInitTexture           ();

    programID_depth = myLoadShader("./shaders/depth_vertex.glsl", "./shaders/depth_fragment.glsl");

    matrixID[0] = glGetUniformLocation (programID_depth, "projectionMat");
    matrixID[1] = glGetUniformLocation (programID_depth, "viewModelMat");
    scalaID[0]  = glGetUniformLocation (programID_depth, "pointRadius");
    scalaID[1]  = glGetUniformLocation (programID_depth, "pointScale");
    scalaID[2]  = glGetUniformLocation (programID_depth, "near");
    scalaID[3]  = glGetUniformLocation (programID_depth, "far");

    programID_normal = myLoadShader("./shaders/normal_vertex.glsl", "./shaders/normal_fragment.glsl");

    matrixID[2] = glGetUniformLocation (programID_normal, "projectionMatInv");

    g_Camera.SetPosition        (g_InitialCameraPosition);
    g_Camera.SetProjectionRH    (45.0f, 1.0f, f_near, f_far);

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_stop);

}

void Renderer::myInitVAO()
{
    glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);
}

void Renderer::myInitVBO()
{
    r3_vertex_buffer_data   = mps.getPos();
    u_vertex_buffer_data    = mps.getType();

    glGenBuffers(1, &vertexBufferPos);
    glGenBuffers(1, &vertexBufferType);
    glGenBuffers(1, &vertexBufferScreen);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferScreen);
    glBufferData(GL_ARRAY_BUFFER, sizeof(r_screen_vertex_buffer_data), r_screen_vertex_buffer_data, GL_STATIC_DRAW);
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


void Renderer::myDrawBuffer_frame()
{
    /*-----draw depth buffer-----*/
    glBindFramebuffer           (GL_FRAMEBUFFER, frameBuffer);

    ///use program
    glClear                     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram                (programID_depth);

    ///send data
    glUniformMatrix4fv          (matrixID[0], 1, GL_FALSE, &g_projectionMat[0][0]);
    glUniformMatrix4fv          (matrixID[1], 1, GL_FALSE, &g_viewModelMat[0][0]);
    glUniform1f                 (scalaID[0], f_pointRadius);
    glUniform1f                 (scalaID[1], f_pointScale);
    glUniform1f                 (scalaID[2], f_near);
    glUniform1f                 (scalaID[3], f_far);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferPos);
    glBufferData                (GL_ARRAY_BUFFER, mps.getNp()*sizeof(*r3_vertex_buffer_data), r3_vertex_buffer_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferType);
    glBufferData                (GL_ARRAY_BUFFER, mps.getNp()*sizeof(*u_vertex_buffer_data), u_vertex_buffer_data, GL_STATIC_DRAW);
    glVertexAttribPointer       (1, 1, GL_UNSIGNED_INT, GL_FALSE, 0, (void*)0);

    ///draw
    glEnableVertexAttribArray   (0);
    glEnableVertexAttribArray   (1);

    glDrawArrays                (GL_POINTS, 0, mps.getNp());

    glDisableVertexAttribArray  (0);
    glDisableVertexAttribArray  (1);
    /*----------------------------*/
}

void Renderer::myDrawBuffer_screen()
{
    /*-----draw to screen-----*/
    glBindFramebuffer           (GL_FRAMEBUFFER, 0);

    ///use program normal
    glClear                     (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram                (programID_normal);

    ///send data
    glUniformMatrix4fv          (matrixID[2], 1, GL_FALSE, &g_projectionMatInv[0][0]);

    glActiveTexture             (GL_TEXTURE0);
    glBindTexture               (GL_TEXTURE_2D, colorTex);
    glUniform1i                 (glGetUniformLocation(programID_normal, "colorTexture"), 0);

    glActiveTexture             (GL_TEXTURE1);
    glBindTexture               (GL_TEXTURE_2D, depthTex);
    glUniform1i                 (glGetUniformLocation(programID_normal, "depthTexture"), 1);

    glBindBuffer                (GL_ARRAY_BUFFER, vertexBufferScreen);
    glVertexAttribPointer       (0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    ///draw
    glEnableVertexAttribArray   (0);

    glDrawArrays                (GL_TRIANGLES, 0, 6);

    glDisableVertexAttribArray  (0);
    /*------------------------*/
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
        case 0x1b:
        {
            stop = true; break;
        }
        case 0x0d:
        {
            stop = false; break;
        }
        case 0x20:
        {
            g_Rotation = glm::angleAxis<float>( -PI * 0.5f, glm::vec3(1, 0, 0) );
            g_Scale = glm::vec3(1.8f);
            g_Pan = glm::vec3(-0.6f, -0.8f, 0.0f);
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
    glClear         (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode    (GL_MODELVIEW);
    glLoadIdentity  ();

    glm::mat4 modelMatrix = glm::translate  (glm::mat4(1.0f), g_Pan )
                          * glm::toMat4     (g_Rotation)
                          * glm::scale      (glm::mat4(1.0f), g_Scale );

    g_viewModelMat      = g_Camera.GetViewMatrix() * modelMatrix;
    g_projectionMat     = g_Camera.GetProjectionMatrix();
    g_projectionMatInv  = glm::inverse(g_projectionMat);

    myDrawBuffer_frame  ();
    myDrawBuffer_screen ();

    glutSwapBuffers     ();

    glutReportErrors    ();

    myFps();
}

void Renderer::myAct()
{
    glutPostRedisplay();

    if(!stop)
    {
        mps.step();
    }

    myDraw();

}

void Renderer::myMainLoop()
{
    glutMainLoop    ();
}

void Renderer::myFinal()
{
    glDeleteBuffers(1, &vertexBufferPos);
    glDeleteBuffers(1, &vertexBufferType);
    glDeleteTextures(1, &depthTex);
    glDeleteTextures(1, &colorTex);

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

        sprintf(_str, "dambreak ->     particles: %d   fps: %4.1f", mps.getNp(), _fps);
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






