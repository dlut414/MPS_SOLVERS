/*LICENCE*/
///sparse matrix in COO format
#ifndef MATRIX_COO_H
#define MATRIX_COO_H

namespace mytype
{
    template <typename I, typename R>
    class matrix_COO
    {
    public:
        matrix_COO(const I n1 = 0, const I n2 = 0) : i_sizeA(n1), i_sizeB(n2)
        {
            r_ptrAij = NULL;
            i_ptrI = NULL;
            i_ptrJ = NULL;
            r_ptrBi = NULL;
        }
        ~matrix_COO()
        {
            r_ptrAij = NULL;
            i_ptrI = NULL;
            i_ptrJ = NULL;
            r_ptrBi = NULL;
        }

    public:
        I i_sizeA;
        I i_sizeB;

        R* r_ptrAij;
        I* i_ptrI;
        I* i_ptrJ;

        R* r_ptrBi;
    };
}
#endif // MATRIX_COO_H
