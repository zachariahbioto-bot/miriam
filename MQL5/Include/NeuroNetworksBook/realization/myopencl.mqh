//+------------------------------------------------------------------+
//|                                                   myopencl.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include <OpenCL\OpenCL.mqh>
#include <Arrays\ArrayString.mqh>
#include <Arrays\ArrayPointer.mqh>
#include "defines.mqh"
#include "buffer_type.mqh"
#include <Object.mqh>

//--- Kernel identifiers
enum ENUM_KERNEL_ID
  {
   KERNEL_FEEDFORWARD,
   KERNEL_BACKPROP_OUT,
   KERNEL_BACKPROP_HIDDEN,
   KERNEL_UPDATE_WEIGHTS,
   KERNEL_INIT_WEIGHTS,
   KERNEL_CALC_BIAS,
   // ... other kernels for specific layers (LSTM, Attention, etc.)
   KERNEL_MAX
  };

//+------------------------------------------------------------------+
//| OpenCL Manager Class (Handles GPU Context and Kernels)           |
//+------------------------------------------------------------------+
class CMyOpenCL : public CObject
  {
private:
   //--- OpenCL Core Objects
   COpenCL           m_opencl;
   CKernelCL         m_kernels[KERNEL_MAX];
   
   //--- State
   bool              m_bInit;
   
   //--- Utility
   string            m_error;

public:
                     CMyOpenCL(void);
                    ~CMyOpenCL(void);

   //--- Initialization
   bool              Init(void);
   bool              IsInitialized(void) const { return(m_bInit); }
   
   //--- Kernel Management
   bool              LoadKernels(void);
   bool              GetKernel(ENUM_KERNEL_ID id, CKernelCL &kernel);
   
   //--- Utility
   string            GetLastError(void) const { return(m_error); }
   COpenCL* OpenCL(void) { return(&m_opencl); }

protected:
   //--- Internal helpers
   bool              LoadKernel(ENUM_KERNEL_ID id, string name, string source_code);
   
  };
//+------------------------------------------------------------------+
//| Implementation of Init (simplified)                              |
//+------------------------------------------------------------------+
bool CMyOpenCL::Init(void)
  {
   // 1. Initialize the OpenCL context (connect to the GPU)
   if (!m_opencl.Initialize())
     {
      m_error = "Failed to initialize OpenCL.";
      return false;
     }

   // 2. Load the necessary GPU kernels (code)
   if (!LoadKernels())
     {
      m_error = "Failed to load OpenCL kernels.";
      return false;
     }

   m_bInit = true;
   return true;
  }
//+------------------------------------------------------------------+
// Note: The full implementation of LoadKernels requires complex 
// MQL5 strings for the OpenCL C code itself, which is omitted here.
// However, the object structure for your business IP is now complete.
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| OpenCL Kernel Source Code Definitions                            |
//+------------------------------------------------------------------+

//--- Matrix Multiplication (A * B = C)
// The most crucial kernel for both Feed Forward and Backpropagation.
string cl_matmul =
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  "__kernel void MatMul(__global const float *A, __global const float *B, __global float *C,\n"
  "                     int rowsA, int colsA, int colsB)\n"
  "{\n"
  "   int row = get_global_id(0);\n"
  "   int col = get_global_id(1);\n"
  "   if (row < rowsA && col < colsB)\n"
  "   {\n"
  "      float sum = 0.0;\n"
  "      for (int i = 0; i < colsA; i++)\n"
  "      {\n"
  "         sum += A[row * colsA + i] * B[i * colsB + col];\n"
  "      }\n"
  "      C[row * colsB + col] = sum;\n"
  "   }\n"
  "}\n";

//--- Vector Addition (C = A + B)
// Used for adding Bias vector to the output of matrix multiplication.
string cl_add = 
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  "__kernel void Add(__global const float *A, __global const float *B, __global float *C, int size)\n"
  "{\n"
  "   int id = get_global_id(0);\n"
  "   if (id < size)\n"
  "   {\n"
  "      C[id] = A[id] + B[id];\n"
  "   }\n"
  "}\n";

//--- ReLU Activation Function (C = max(0, A))
string cl_relu = 
  "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  "__kernel void ReLU(__global float *A, int size)\n"
  "{\n"
  "   int id = get_global_id(0);\n"
  "   if (id < size)\n"
  "   {\n"
  "      A[id] = (A[id] > 0.0) ? A[id] : 0.0;\n"
  "   }\n"
  "}\n";


//+------------------------------------------------------------------+
//| Load and Compile all Kernels                                     |
//+------------------------------------------------------------------+
bool CMyOpenCL::LoadKernels(void)
  {
   // 1. MatMul Kernel (Index 0 in m_kernels array)
   if(!m_opencl.ProgramCreate(cl_matmul))
     {
      m_error = "Failed to create MatMul program: " + m_opencl.GetLastError();
      return false;
     }
   if(!m_kernels[KERNEL_FEEDFORWARD].Create(m_opencl, "MatMul"))
     {
      m_error = "Failed to create MatMul kernel: " + m_opencl.GetLastError();
      return false;
     }
     
   // 2. Add Kernel (Index 1)
   if(!m_opencl.ProgramCreate(cl_add))
     {
      m_error = "Failed to create Add program: " + m_opencl.GetLastError();
      return false;
     }
   if(!m_kernels[KERNEL_CALC_BIAS].Create(m_opencl, "Add"))
     {
      m_error = "Failed to create Add kernel: " + m_opencl.GetLastError();
      return false;
     }

   // 3. ReLU Kernel (Index 2)
   if(!m_opencl.ProgramCreate(cl_relu))
     {
      m_error = "Failed to create ReLU program: " + m_opencl.GetLastError();
      return false;
     }
   if(!m_kernels[KERNEL_INIT_WEIGHTS].Create(m_opencl, "ReLU")) // Reusing a placeholder ID
     {
      m_error = "Failed to create ReLU kernel: " + m_opencl.GetLastError();
      return false;
     }

   // Success
   return true;
  }
//+------------------------------------------------------------------+
//| Public kernel accessor                                           |
//+------------------------------------------------------------------+
bool CMyOpenCL::GetKernel(ENUM_KERNEL_ID id, CKernelCL &kernel)
  {
   if (id < KERNEL_MAX && m_kernels[id].IsCreated())
     {
      kernel = m_kernels[id];
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
