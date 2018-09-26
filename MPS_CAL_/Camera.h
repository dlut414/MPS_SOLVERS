/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Published under CC BY-NC
*/
//Camera.h
///defination of class Camera
#ifndef CAMERA_H
#define CAMERA_H

#include "header.h"

class Camera
{
public:

    Camera()
        : m_Viewport(0)
        , m_Position(0)
        , m_Rotation()
        , m_ViewMatrix(1)
        , m_ProjectionMatrix(1)
        , m_ViewDirty(false)
    {}
    Camera( int screenWidth, int screenHeight )
        : m_Viewport( 0, 0, screenWidth, screenHeight )
        , m_Position(0)
        , m_Rotation()
        , m_ViewMatrix(1)
        , m_ProjectionMatrix(1)
        , m_ViewDirty( false )
    {}

    void SetViewport( int x, int y, int width, int height )
    {
        m_Viewport = glm::vec4( x, y, width, height );
        // Requires gl.h
        glViewport(x, y, width, height);
    }
    glm::vec4 GetViewport() const {return m_Viewport;}

    void SetProjectionRH( float fov, float aspectRatio, float zNear, float zFar )
    {
        m_ProjectionMatrix = glm::perspective( glm::radians(fov), aspectRatio, zNear, zFar );
    }

    void ApplyViewMatrix() {UpdateViewMatrix();}

    void SetPosition( const glm::vec3& pos )
    {
        m_Position = pos;
        m_ViewDirty = true;
    }
    glm::vec3 GetPosition() const {return m_Position;}

    // Translate the camera by some amount. If local is TRUE (default) then the translation should
    // be applied in the local-space of the camera. If local is FALSE, then the translation is
    // applied in world-space.
    void Translate( const glm::vec3& delta, bool local = true )
    {
        if ( local )
        {
            m_Position += m_Rotation * delta;
        }
        else
        {
            m_Position += delta;
        }
        m_ViewDirty = true;
    }

    void SetRotation( const glm::quat& rot )
    {
        m_Rotation = rot;
        m_ViewDirty = true;
    }
    glm::quat GetRotation() const {return m_Rotation;}


    void SetEulerAngles( const glm::vec3& eulerAngles )
    {
        m_Rotation = glm::quat(glm::radians(eulerAngles));
    }
    glm::vec3 GetEulerAngles() const {return glm::degrees(glm::eulerAngles( m_Rotation ));}

    // Rotate the camera by some amount.
    void Rotate( const glm::quat& rot )
    {
        m_Rotation = m_Rotation * rot;
        m_ViewDirty = true;
    }

    glm::mat4 GetProjectionMatrix() {return m_ProjectionMatrix;}
    glm::mat4 GetViewMatrix()
    {
        UpdateViewMatrix();
        return m_ViewMatrix;
    }

protected:

    void UpdateViewMatrix()
    {
        if ( m_ViewDirty )
        {
            glm::mat4 translate = glm::translate(-m_Position);
            // Since we know the rotation matrix is orthonormalized, we can simply
            // transpose the rotation matrix instead of inversing.
            glm::mat4 rotate = glm::transpose(glm::toMat4(m_Rotation));

            m_ViewMatrix = rotate * translate;

            m_ViewDirty = false;
        }
    }

    glm::vec4 m_Viewport;

    glm::vec3 m_Position;
    glm::quat m_Rotation;

    glm::mat4 m_ViewMatrix;
    glm::mat4 m_ProjectionMatrix;

private:
    bool m_ViewDirty;
};


#endif // CAMERA_H
