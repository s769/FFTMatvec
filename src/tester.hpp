#ifndef __TESTER_HPP__
#define __TESTER_HPP__

#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

namespace Tester {

void checkOnesMatvec(Comm& comm, Matrix& mat, Vector& out, bool conj, bool full);

}


#endif // __TESTER_HPP__