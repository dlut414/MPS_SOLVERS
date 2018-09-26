/*-------------------------------------------
PROGRAM CHANGING OUTPUT FILES INTO VTK FILES
--------------------------------------------*/
#include <iostream>
#include "MPS_POST.h"
using namespace std;

int main()
{
    MPS_POST output;
    output.LoadCase();
    output.WriteData();
    return 0;
}
