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
