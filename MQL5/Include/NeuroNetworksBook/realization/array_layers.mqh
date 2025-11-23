//+------------------------------------------------------------------+
//|                                              array_layers.mqh    |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

#include <Arrays\ArrayObj.mqh>
#include "defines.mqh"
#include "layer_description.mqh"

//--- Forward declarations of neuron classes (we will build these next)
class CNeuronBase;
class CNeuronConv;
class CNeuronProof;
class CNeuronLSTM;
class CNeuronAttention;
class CNeuronMHAttention;
class CNeuronGPT;
class CNeuronDropout;
class CNeuronBatchNorm;
class CMyOpenCL;

//+------------------------------------------------------------------+
//| Dynamic storage array of neural layers                           |
//+------------------------------------------------------------------+
class CArrayLayers : public CArrayObj
  {
protected:
   CMyOpenCL* m_cOpenCL;        // Pointer to GPU context
   int               m_iFileHandle;    // File handle for loading operations

public:
                     CArrayLayers(void) : m_cOpenCL(NULL), m_iFileHandle(INVALID_HANDLE) { }
                    ~CArrayLayers(void) { }

   //--- OpenCL Control
   virtual bool      SetOpencl(CMyOpenCL *opencl);

   //--- File Operations
   virtual bool      Load(const int file_handle) override;

   //--- Factory Methods (Creating layers)
   virtual bool      CreateElement(const int index) override;
   virtual bool      CreateElement(const int index, CLayerDescription* description);

   //--- Identification
   virtual int       Type(void) const { return(defArrayLayers); }
  };
//+------------------------------------------------------------------+
//| Load from file override (saves handle for CreateElement)         |
//+------------------------------------------------------------------+
bool CArrayLayers::Load(const int file_handle)
  {
   m_iFileHandle = file_handle;
   return CArrayObj::Load(file_handle);
  }
//+------------------------------------------------------------------+
//| Factory Method: Create layer from file ID                        |
//+------------------------------------------------------------------+
bool CArrayLayers::CreateElement(const int index)
  {
   if(index < 0 || m_iFileHandle == INVALID_HANDLE)
      return false;

   if(!Reserve(index + 1))
      return false;

   // Read the layer type from the file
   int type = FileReadInteger(m_iFileHandle);
   CNeuronBase *temp = NULL;

   // Switch factory to create the correct object
   switch(type)
     {
      case defNeuronBase:        temp = new CNeuronBase();        break;
      case defNeuronConv:        temp = new CNeuronConv();        break;
      case defNeuronProof:       temp = new CNeuronProof();       break;
      case defNeuronLSTM:        temp = new CNeuronLSTM();        break;
      case defNeuronAttention:   temp = new CNeuronAttention();   break;
      case defNeuronMHAttention: temp = new CNeuronMHAttention(); break;
      case defNeuronGPT:         temp = new CNeuronGPT();         break;
      case defNeuronDropout:     temp = new CNeuronDropout();     break;
      case defNeuronBatchNorm:   temp = new CNeuronBatchNorm();   break;
      default: return false;
     }

   if(!temp)
      return false;

   // Clean up old element if exists
   if(m_data[index])
      delete m_data[index];

   // Set GPU context and save to array
   temp.SetOpenCL(m_cOpenCL);
   m_data[index] = temp;

   return true;
  }
//+------------------------------------------------------------------+
//| Factory Method: Create layer from Description                    |
//+------------------------------------------------------------------+
bool CArrayLayers::CreateElement(const int index, CLayerDescription *desc)
  {
   if(index < 0 || !desc)
      return false;

   if(!Reserve(index + 1))
      return false;

   CNeuronBase *temp = NULL;

   switch(desc.type)
     {
      case defNeuronBase:        temp = new CNeuronBase();        break;
      case defNeuronConv:        temp = new CNeuronConv();        break;
      case defNeuronProof:       temp = new CNeuronProof();       break;
      case defNeuronLSTM:        temp = new CNeuronLSTM();        break;
      case defNeuronAttention:   temp = new CNeuronAttention();   break;
      case defNeuronMHAttention: temp = new CNeuronMHAttention(); break;
      case defNeuronGPT:         temp = new CNeuronGPT();         break;
      case defNeuronDropout:     temp = new CNeuronDropout();     break;
      case defNeuronBatchNorm:   temp = new CNeuronBatchNorm();   break;
      default: return false;
     }

   if(!temp)
      return false;

   // Initialize the layer with the description
   if(!temp.Init(desc))
     {
      delete temp;
      return false;
     }

   if(m_data[index])
      delete m_data[index];

   temp.SetOpenCL(m_cOpenCL);
   m_data[index] = temp;
   m_data_total  = (int)fmax(m_data_total, index + 1);

   return true;
  }
//+------------------------------------------------------------------+
//| Distribute OpenCL context to all layers                          |
//+------------------------------------------------------------------+
bool CArrayLayers::SetOpencl(CMyOpenCL *opencl)
  {
   if(m_cOpenCL)
      delete m_cOpenCL;
   m_cOpenCL = opencl;

   for(int i = 0; i < m_data_total; i++)
     {
      if(!m_data[i])
         return false;
      // Cast to CNeuronBase to call SetOpenCL
      if(!((CNeuronBase *)m_data[i]).SetOpenCL(m_cOpenCL))
         return false;
     }
   return(!!m_cOpenCL);
  }
//+------------------------------------------------------------------+
