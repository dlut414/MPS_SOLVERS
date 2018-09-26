#include <iostream>
#include <cstdio>
#include <vector>
using namespace std;
const static double L = 0.26;
const static double EPS = 1e-10;
struct double3
{
    double x;
    double y;
    double z;
};
int main()
{
    unsigned np = 0;
    unsigned nb2 = 0;
    unsigned nb1 = 0;
    unsigned nfluid = 0;

    int dim = 0;
    double dp = 0.0;
    double niu = 0.0;
    double rho = 0.0;

    unsigned length;
    unsigned height;
    unsigned depth;

    char buffer[256];
    FILE* fid = fopen("input.dat","r");

    fgets(buffer, sizeof(buffer), fid);
    fgets(buffer, sizeof(buffer), fid);
    sscanf(buffer, "%d", &dim);
    fgets(buffer, sizeof(buffer), fid);
    fgets(buffer, sizeof(buffer), fid);
    sscanf(buffer, "%lf", &rho);
    fgets(buffer, sizeof(buffer), fid);
    fgets(buffer, sizeof(buffer), fid);
    sscanf(buffer, "%lf", &niu);
    fgets(buffer, sizeof(buffer), fid);
    fgets(buffer, sizeof(buffer), fid);
    sscanf(buffer, "%lf", &dp);

    fclose(fid);

    cout << "dim: " << dim << endl;
    cout << "dp:  " << dp << endl;
    /*-----damebreak-----*/
    length = unsigned(2 * L / dp) + 5;
    depth  = unsigned(2 * L / dp) + 5;
    height = unsigned(2 * L / dp) + 5;
    /*-------------------*/

    /*-----laminar-----*/
    /*
    row = 4 + int(4 * L / dp) + 1;
    line = int(L / dp) + 1 + 2 + 2;
    */
    /*-------------------*/

    //from left to right
    vector<double3> nb2_list;
    vector<double3> nb1_list;
    vector<double3> fluid_list;
    double3 tmp = {0,0,0};
    for(unsigned i=0;i<height;i++)
    {
        for(unsigned j=0;j<depth;j++)
        {
            for(unsigned k=0;k<length;k++)
            {
                tmp.x = k * dp;
                tmp.y = j * dp;
                tmp.z = i * dp;
                /*-----double dambreak-----*/
                if(i < 2 || i > height - 3 || j < 2 || j > depth - 3 || k < 2 || k > length - 3)
                {
                    nb2++;
                    nb2_list.push_back(tmp);
                }
                else if( i == 2 || i == height - 3 || j == 2 || j == depth - 3 || k == 2 || k == length - 3 )
                {
                    nb1++;
                    nb1_list.push_back(tmp);
                }
                else if((tmp.x <= L-2*dp && tmp.y > L+4*dp && tmp.z <= L) || (tmp.x > L+5*dp && tmp.y <= L-4*dp  && tmp.z <= L))
                {
                    nfluid++;
                    fluid_list.push_back(tmp);
                }
                else continue;
                /*------------------*/
                /*-----dambreak-----*/
                /*
                if(i < 2 || i > height - 3 || j < 2 || j > depth - 3 || k < 2 || k > length - 3)
                {
                    nb2++;
                    nb2_list.push_back(tmp);
                }
                else if( i == 2 || i == height - 3 || j == 2 || j == depth - 3 || k == 2 || k == length - 3 )
                {
                    nb1++;
                    nb1_list.push_back(tmp);
                }
                else if(tmp.x <= (L + dp * 2) && tmp.z <= L + 2 * dp)
                {
                    nfluid++;
                    fluid_list.push_back(tmp);
                }
                else continue;
                */
                /*------------------*/
                /*-----droplet-----*/
                /*
                if(i < 2 || i > height - 3 || j < 2 || j > depth - 3 || k < 2 || k > length - 3)
                {
                    nb2++;
                    nb2_list.push_back(tmp);
                }
                else if( i == 2 || i == height - 3 || j == 2 || j == depth - 3 || k == 2 || k == length - 3 )
                {
                    nb1++;
                    nb1_list.push_back(tmp);
                }
                else if(tmp.z <= 0.5f*L + 2 * dp)
                {
                    nfluid++;
                    fluid_list.push_back(tmp);
                }
                else if(tmp.z >= 0.75 * L && tmp.z <= 1.25f * L && tmp.x >= 6.f/8.f*L && tmp.x <= 10.f/8.f*L && tmp.y >= 6.f/8.f*L && tmp.y <= 10.f/8.f*L)
                {
                    nfluid++;
                    fluid_list.push_back(tmp);
                }
                else continue;
                */
                /*------------------*/
                /*-----lanimara-----*/
                /*
                if(i < 2 || i > line - 3)
                {
                    nb2++;
                    nb2_list.push_back(tmp);
                }
                else if( i == 2 || i == line - 3)
                {
                    nb1++;
                    nb1_list.push_back(tmp);
                }
                else if(i > 2 && i < line - 3)
                {
                    nfluid++;
                    fluid_list.push_back(tmp);
                }
                else continue;
                */
                /*------------------*/
            }
        }
    }

    np = nb2 + nb1 + nfluid;

    cout << "np:  " << np << endl;

    fid = fopen("GEO.dat","wt");
    fprintf(fid, "%d %d %d\n", np, nb2, nb1);
    for(unsigned i=0;i<nb2;i++)
    {
        fprintf(fid, "%lf %lf %lf ", nb2_list[i].x, nb2_list[i].y, nb2_list[i].z);
    }
    for(unsigned i=0;i<nb1;i++)
    {
        fprintf(fid, "%lf %lf %lf ", nb1_list[i].x, nb1_list[i].y, nb1_list[i].z);
    }
    for(unsigned i=0;i<nfluid;i++)
    {
        fprintf(fid, "%lf %lf %lf ", fluid_list[i].x, fluid_list[i].y, fluid_list[i].z);
    }

    //velocity
    for(unsigned i=0;i<np;i++)
    {
        fprintf(fid, "0.0 0.0 0.0 ");
    }

    //pressure
    for(unsigned i=0;i<np;i++)
    {
        fprintf(fid, "0.0 ");
    }

    fclose(fid);
    return 0;
}
