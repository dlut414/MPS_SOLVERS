/*-------------------------------------------
PROGRAM CHANGING OUTPUT FILES INTO VTK FILES
--------------------------------------------*/
#include "MPS_POST.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "common.h"

MPS_POST::MPS_POST(){}
MPS_POST::~MPS_POST()
{
    delete[] id;

    delete[] posx;
    delete[] posy;
    delete[] posz;

    delete[] velx;
    delete[] vely;
    delete[] velz;

    delete[] pres;

    delete[] cellList;
    delete[] cellType;
    delete[] neighbor;
}

void MPS_POST::LoadCase()
{
    char buffer[256];
    char _file_name[256];

    strcpy(_file_name, file_name);
    strcat(_file_name, "input.dat");

    fid_read = fopen(_file_name,"r");

    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    sscanf(buffer, "%lf", &dp);
    printf("dp:   %lf\n", dp);

    fgets(buffer, sizeof(buffer), fid_read);
    fgets(buffer, sizeof(buffer), fid_read);
    sscanf(buffer,"%lf", &r0);
    printf("r0:   %lf\n", r0);
    fclose(fid_read);

    strcpy(_file_name, file_name);
    strcat(_file_name, "GEO.dat");
    fid_read = fopen(_file_name,"r");
    fscanf(fid_read, "%d %d %d", &np, &nb2, &nb1);
    fclose(fid_read);

    cellType = new int[np];

    neighbor = new int*[np];
    for(int i=0;i<np;i++)
    {
        neighbor[i] = new int[2];
    }

    id = new int[np];
    posx = new double[np];
    posy = new double[np];
    posz = new double[np];

    velx = new double[np];
    vely = new double[np];
    velz = new double[np];

    pres = new double[np];
}

void MPS_POST::LoadData()
{
    for(int i=0;i<np;i++)
    {
        fscanf(fid_read, "%lf %lf %lf %lf %lf %lf %lf %d",
        &posx[i], &posy[i], &posz[i], &velx[i], &vely[i], &velz[i], &pres[i], &id[i]);
    }
}

void MPS_POST::WriteHead()
{
    fprintf(fid_write, "# vtk DataFile Version 3.0\n");
    fprintf(fid_write, "unstructured grid\n");
    fprintf(fid_write, "ASCII\n");
    fprintf(fid_write, "DATASET UNSTRUCTURED_GRID\n");
}

void MPS_POST::WritePoint()
{
    fprintf(fid_write, "POINTS %d float\n",np);
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%lf %lf %lf ", posx[i], posy[i], posz[i]);
    }
    fprintf(fid_write, "\n");
}

void MPS_POST::WriteCell1()
{
    fprintf(fid_write, "CELLS %d %d\n",np , 2*np);
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%d %d ", 1, i);
    }
    fprintf(fid_write, "\n");

    fprintf(fid_write, "CELL_TYPES %d\n", np);
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "1 ");
    }
    fprintf(fid_write, "\n");
}

void MPS_POST::WriteCell5()
{
    int _num_cell1 = 0;
    int _num_cell3 = 0;
    int _num_cell5 = 0;
    double rr;
    double r02 = (dp * r0) * (dp * r0);
    double tmp[2];

    for(int i=0;i<np;i++)
    {
        cellType[i] = 1;
        neighbor[i][0] = neighbor[i][1] = i;
    }

    /*-----make cells-----*/
    for(int i=0;i<np;i++)
    {
        tmp[0] = tmp[1] = 1e5;
        for(int j=0;j<np;j++)
        {
            if(j == i) continue;
            rr = pow(posx[i] - posx[j],2) + pow(posy[i] - posy[j],2) + pow(posz[i] - posz[j],2);

            if(rr > r02) continue;
            if(rr < tmp[0])
            {
                tmp[0] = rr;
                neighbor[i][0] = j;
            }
            else if(rr < tmp[1])
            {
                tmp[1] = rr;
                neighbor[i][1] = j;
            }
        }
        if(neighbor[i][0] == i)
        {
            cellType[i] = 1;
            _num_cell1++;
        }
        else if(neighbor[i][0] != i && neighbor[i][1] == i)
        {
            cellType[i] = 3;
            _num_cell3++;
        }
        else if(neighbor[i][0] != i && neighbor[i][1] != i)
        {
            cellType[i] = 5;
            _num_cell5++;
        }
    }
    /*--------------------*/

    fprintf(fid_write, "CELLS %d %d\n",_num_cell1+_num_cell3+_num_cell5 , _num_cell1*2 + 1+_num_cell3*3 + 1+_num_cell5*4);

    for(int i=0;i<np;i++)
    {
        if(cellType[i] == 1) fprintf(fid_write, "%d %d ", 1, i);
        else if(cellType[i] == 3) fprintf(fid_write, "%d %d %d ", 2, i, neighbor[i][0]);
        else if(cellType[i] == 5) fprintf(fid_write, "%d %d %d %d ", 3, i, neighbor[i][0], neighbor[i][1]);
    }
    fprintf(fid_write, "\n");

    fprintf(fid_write, "CELL_TYPES %d\n", np);
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%d ", cellType[i]);
    }
    fprintf(fid_write, "\n");
}

void MPS_POST::WriteAttribute()
{
    fprintf(fid_write, "POINT_DATA %d\n", np);

    fprintf(fid_write, "SCALARS id int 1\n");
    fprintf(fid_write, "LOOKUP_TABLE default\n");
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%d ", id[i]);
    }
    fprintf(fid_write, "\n");

    fprintf(fid_write, "SCALARS pressure float 1\n");
    fprintf(fid_write, "LOOKUP_TABLE default\n");
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%lf ", pres[i]);
    }
    fprintf(fid_write, "\n");

    fprintf(fid_write, "VECTORS velocity float\n");
    for(int i=0;i<np;i++)
    {
        fprintf(fid_write, "%lf %lf %lf ", velx[i], vely[i], velz[i]);
    }
    fprintf(fid_write, "\n");
}

void MPS_POST::WriteData()
{
    char _data_file[256];
    char _out_file[256];
    c = 0;
    do
    {
        strcpy(data_file, file_name);
        strcpy(out_file, file_name);
        sprintf(_data_file, "out/%04d.out",c);
        sprintf(_out_file, "out/part%04d.vtk",c);
        strcat(data_file, _data_file);
        strcat(out_file, _out_file);

        fid_read = fopen(data_file, "r");
        if(fid_read == NULL) break;
        fid_write = fopen(out_file, "wt");

        LoadData();
        WriteHead();

        WritePoint();
        WriteCell1();
        WriteAttribute();

        fclose(fid_read);
        fclose(fid_write);
        printf("successfully saved file %04d\n",c);
    }while(++c);
}
