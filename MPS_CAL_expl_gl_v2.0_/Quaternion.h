/*
LICENCE
*/
//Quaternion.h
///defination of class Quaternion
#ifndef QUATERNION_H
#define QUATERNION_H

template <typename T>
class Quaternion
{

public:
    ~Quaternion();
    Quaternion()
    {
        q[0] = 1.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
    }
    Quaternion(Quaternion& x)
    {
        q[0] = x[0];
        q[1] = x[1];
        q[2] = x[2];
        q[3] = x[3];
    }

    Quaternion operator+ (const Quaternion& x)
    {
        Quaternion tmp;
        tmp.q[0] = this.q[0] + x.q[0];
        tmp.q[1] = this.q[1] + x.q[1];
        tmp.q[2] = this.q[2] + x.q[2];
        tmp.q[3] = this.q[3] + x.q[3];
        return tmp;
    }
    Quaternion operator- (const Quaternion& x)
    {
        Quaternion tmp;
        tmp.q[0] = this.q[0] - x.q[0];
        tmp.q[1] = this.q[1] - x.q[1];
        tmp.q[2] = this.q[2] - x.q[2];
        tmp.q[3] = this.q[3] - x.q[3];
        return tmp;
    }
    Quaternion operator* (const Quaternion& x)
    {
        Quaternion tmp;
        tmp.q[0] = this.q[0]*x.q[0] - this.q[1]*x.q[1] - this.q[2]*x.q[2] - this.q[3]*x.q[3];
        tmp.q[1] = this.q[0]*x.q[1] + x.q[0]*this.q[1] + this.q[2]*x.q[3] - x.q[2]*this.q[3];
        tmp.q[2] = this.q[0]*x.q[2] + x.q[0]*this.q[2] + this.q[3]*x.q[1] - x.q[3]*this.q[1];
        tmp.q[3] = this.q[0]*x.q[3] + x.q[0]*this.q[3] + this.q[1]*x.q[2] - x.q[1]*this.q[2];
        return tmp;
    }

private:
    T q[4];

};

#endif // QUATERNION_H
