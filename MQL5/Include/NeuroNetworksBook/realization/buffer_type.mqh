//+------------------------------------------------------------------+
//|                                                buffer_type.mqh   |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include "defines.mqh"
#include <Object.mqh>
#include <OpenCL\OpenCL.mqh>

//--- Forward declaration for OpenCL manager (we build this later)
class CMyOpenCL;

//+------------------------------------------------------------------+
//| Class for storing weights, biases, and data (input/output)       |
//+------------------------------------------------------------------+
class CBufferType : public CObject
  {
private:
   //--- Data storage
   MATRIX            m_buffer;

   //--- OpenCL (GPU) data
   CMyOpenCL* m_cOpenCL;
   CBufferCL* m_cl_buffer;
   bool              m_bOpenCL;
   bool              m_bUseOpenCL;

public:
                     CBufferType(void);
                    ~CBufferType(void);

   //--- Data Initialization
   bool              Create(uint rows, uint cols, TYPE value = 0);
   bool              Create(uint size, TYPE value = 0);

   //--- Accessors
   MATRIX            Buffer(void) const { return(m_buffer); }
   TYPE              Get(uint row, uint col) const { return(m_buffer[row][col]); }
   void              Set(uint row, uint col, TYPE value) { m_buffer[row][col] = value; }
   uint              Rows(void) const { return(m_buffer.Rows()); }
   uint              Cols(void) const { return(m_buffer.Cols()); }

   //--- OpenCL Control
   bool              InitOpenCL(void);
   void              UseOpenCL(bool value);
   bool              UseOpenCL(void) const { return(m_bUseOpenCL); }
   bool              SetOpenCL(CMyOpenCL *opencl);
   bool              LoadToOpenCL(void);
   bool              ReadFromOpenCL(void);

   //--- Utility
   void              Zero(void);
   void              FillRandom(TYPE min, TYPE max);

   //--- Identification
   virtual int       Type(void) const { return(defBuffer); }

   //--- Persistence
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
  };
//+------------------------------------------------------------------+
//| Constructor/Destructor                                           |
//+------------------------------------------------------------------+
CBufferType::CBufferType(void) : m_cOpenCL(NULL), m_cl_buffer(NULL), m_bOpenCL(false), m_bUseOpenCL(false)
  {
  }

CBufferType::~CBufferType(void)
  {
   if(m_cl_buffer)
     {
      delete m_cl_buffer;
      m_cl_buffer = NULL;
     }
  }

bool CBufferType::Create(uint rows, uint cols, TYPE value)
  {
   if (rows == 0 || cols == 0) return false;
   m_buffer.Resize(rows, cols);
   m_buffer.Fill(value);
   return true;
  }

bool CBufferType::Create(uint size, TYPE value)
  {
   // Vector case
   return Create(1, size, value);
  }

void CBufferType::Zero(void)
  {
   m_buffer.Zeros();
  }
// Note: Other method implementations (Save, Load, OpenCL) are complex and will be built in the future.
// The structure is defined for now.
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Set OpenCL Context                                               |
//+------------------------------------------------------------------+
bool CBufferType::SetOpenCL(CMyOpenCL *opencl)
  {
   if(!opencl)
      return false;
   m_cOpenCL = opencl;
   m_bOpenCL = m_cOpenCL.IsInitialized();
   return true;
  }
//+------------------------------------------------------------------+
//| Initialize GPU Memory Buffer                                     |
//+------------------------------------------------------------------+
bool CBufferType::InitOpenCL(void)
  {
   if(!m_bOpenCL)
      return false;

   // 1. Calculate buffer size in bytes
   uint size = m_buffer.Size();
   if(size == 0)
      return false;
   
   // 2. Create the OpenCL buffer object
   m_cl_buffer = new CBufferCL();
   if(m_cl_buffer == NULL)
      return false;

   // 3. Allocate memory on the GPU (read/write access)
   if(!m_cl_buffer.BufferCreate(m_cOpenCL.OpenCL(), size * sizeof(TYPE), CL_MEM_READ_WRITE))
     {
      delete m_cl_buffer;
      m_cl_buffer = NULL;
      return false;
     }
     
   m_bUseOpenCL = true;
   return true;
  }
//+------------------------------------------------------------------+
//| Load data from CPU to GPU (Upload Weights/Input)                 |
//+------------------------------------------------------------------+
bool CBufferType::LoadToOpenCL(void)
  {
   if(!m_bUseOpenCL || m_cl_buffer == NULL)
      return false;
   
   // Map the CPU matrix data directly to the GPU buffer
   return m_buffer.BufferWrite(m_cl_buffer);
  }
//+------------------------------------------------------------------+
//| Read data from GPU to CPU (Download Predictions/Gradients)       |
//+------------------------------------------------------------------+
bool CBufferType::ReadFromOpenCL(void)
  {
   if(!m_bUseOpenCL || m_cl_buffer == NULL)
      return false;
      
   // Map the GPU buffer data back to the CPU matrix
   return m_buffer.BufferRead(m_cl_buffer);
  }
//+------------------------------------------------------------------+
