//+------------------------------------------------------------------+
//|                                                      defines.mqh |
//|                                   Copyright 2025, Your Name Ltd. |
//|                                      Neural Network Business IP  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Name Ltd."
#property link      "https://www.mql5.com"

//--- Object Identifiers (The DNA of our system)
#define defNeuronNet        0x8000
#define defArrayLayers      0x8001
#define defBuffer           0x8002
#define defActivation       0x8003
#define defLayerDescription 0x8004

//--- Neuron Types
#define defNeuronBase       0x8010
#define defNeuronConv       0x8011
#define defNeuronProof      0x8012
#define defNeuronLSTM       0x8013
#define defNeuronAttention  0x8014
#define defNeuronMHAttention 0x8015
#define defNeuronGPT        0x8016
#define defNeuronDropout    0x8017
#define defNeuronBatchNorm  0x8018

//--- Optimization Methods
enum ENUM_OPTIMIZATION
  {
   None=-1,
   SGD,
   MOMENTUM,
   AdaGrad,
   RMSProp,
   AdaDelta,
   Adam
  };

//--- Data Type Macros (For switching between double/float for GPU)
#define TYPE   double
#define MATRIX matrix<TYPE>
#define VECTOR vector<TYPE>

//--- Default Hyperparameters
#define defLossSmoothFactor 1000
#define defLearningRate     (TYPE)3.0e-4
#define defBeta1            (TYPE)0.9
#define defBeta2            (TYPE)0.999
#define defLambdaL1         (TYPE)0
#define defLambdaL2         (TYPE)0
