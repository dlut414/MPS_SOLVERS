/*
 * LICENCE
 * copyright 2014 ~ ****
 * Some rights reserved.
 * Author: HUFANGYUAN
 * Released under CC BY-NC
*/
#ifndef BITMAP_H
#define BITMAP_H

#include <cstdio>

typedef unsigned char  U8;
typedef unsigned short U16;
typedef unsigned int   U32;

#pragma pack(push, 1)

struct BITMAPFILEHEADER
{
    U16 bfType;
    U32 bfSize;
    U16 bfReserved1;
    U16 bfReserved2;
    U32 bfOffBits;
};

struct BITMAPINFOHEADER
{
    U32 biSize;
    U32 biWidth;
    U32 biHeight;
    U16 biPlanes;
    U16 biBitCount;
    U32 biCompression;
    U32 biSizeImage;
    U32 biXPelsPerMeter;
    U32 biYPelsPerMeter;
    U32 biClrUsed;
    U32 biClrImportant;
};

struct RGBQUAD
{
    U8 rgbBlue;
    U8 rgbGreen;
    U8 rgbRed;
    U8 rgbReserved;
};

#pragma pack(pop)

class Bitmap
{
public:
    Bitmap() {}
    Bitmap(unsigned w, unsigned h) : width(w), height(h)
    {
        data = new unsigned char [w*h];
    }
    ~Bitmap()
    {
        if(!data) delete[] data;
    }

    bool readBmp(const char* path)
    {
        FILE* fid = fopen(path, "rb");
        if( !fid )
        {
            printf("no file exists !\n");
            return 0;
        }

        if( fread(header, 1, 54, fid) != 54 )
        {
            printf("wrong header, or not bmp !\n");
            return 0;
        }

        if( header[0] != 'B' || header[1] != 'M' )
        {
            printf("wrong header, no BM heading !\n");
            return 0;
        }

        dataPos     = *(unsigned*)&(header[0x0A]);
        width       = *(unsigned*)&(header[0x12]);
        height      = *(unsigned*)&(header[0x16]);
        imageSize   = *(unsigned*)&(header[0x22]);

        if(imageSize == 0) imageSize = 3 * width * height;
        if(dataPos == 0) dataPos = 54;

        data = new unsigned char[imageSize];
        fread(data, 1, imageSize, fid);

        fclose(fid);

        return 1;
    }

    bool writeBmp(const char* path)
    {
        writeHeader();

        FILE* fid = fopen(path, "wb");
        if( !fid )
        {
            printf("file creation failed !\n");
            return 0;
        }

        fwrite(header, 1, 54, fid);
        fwrite(data, 1, imageSize, fid);

        fclose(fid);

        return 1;
    }

    void SaveAsBMP(const char *fileName)
    {
        FILE *file;
        unsigned long imageSize;
        GLbyte *data=NULL;
        GLint viewPort[4];
        GLenum lastBuffer;
        BITMAPFILEHEADER bmfh;
        BITMAPINFOHEADER bmih;
        bmfh.bfType = 0x4D42;
        bmfh.bfReserved1 = 0;
        bmfh.bfReserved2 = 0;
        bmfh.bfOffBits = 54;
        glGetIntegerv(GL_VIEWPORT, viewPort);
        imageSize = ((viewPort[2]+((4-(viewPort[2]%4))%4))*viewPort[3]*3)+2;
        bmfh.bfSize = imageSize + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
        data=(GLbyte*)malloc(imageSize);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        glPixelStorei(GL_PACK_SKIP_ROWS, 0);
        glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_PACK_SWAP_BYTES, 1);
        glGetIntegerv(GL_READ_BUFFER, (GLint*)&lastBuffer);
        glReadBuffer(GL_FRONT);
        glReadPixels(0, 0, viewPort[2], viewPort[3], GL_BGR, GL_UNSIGNED_BYTE, data);
        data[imageSize-1] = 0;
        data[imageSize-2] = 0;
        glReadBuffer(lastBuffer);
        file = fopen(fileName,"wb");
        bmih.biSize = 40;
        bmih.biWidth = viewPort[2];
        bmih.biHeight = viewPort[3];
        bmih.biPlanes = 1;
        bmih.biBitCount = 24;
        bmih.biCompression = 0;
        bmih.biSizeImage = imageSize;
        bmih.biXPelsPerMeter = 45089;
        bmih.biYPelsPerMeter = 45089;
        bmih.biClrUsed = 0;
        bmih.biClrImportant = 0;
        fwrite(&bmfh, sizeof(bmfh), 1, file);
        fwrite(&bmih, sizeof(bmih), 1, file);
        fwrite(data,imageSize, 1, file);
        free(data);
        fclose(file);
    }

public:
    unsigned int dataPos;
    unsigned int width, height;
    unsigned int imageSize; ///width * height * 3 (rgb)
    unsigned char* data;

private:
    void writeHeader()
    {
        imageSize = 3 * width * height;
        dataPos = 54;

        for(int i=0;i<54;i++) header[i] = 0x00;

        header[0] = 'B';
        header[1] = 'M';

        *(unsigned*)&(header[0x0A]) = dataPos;
        *(unsigned*)&(header[0x12]) = width;
        *(unsigned*)&(header[0x16]) = height;
        *(unsigned*)&(header[0x22]) = imageSize;
    }

private:
    unsigned char header[54];

};

#endif // BITMAP_H
