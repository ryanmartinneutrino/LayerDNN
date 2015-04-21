#ifndef __LAYERDNN_TYPES_H__
#define __LAYERDNN_TYPES_H__

#include <iostream>
#include <stdlib.h>
#include <vector>


//For error handling, PRETTY_FUNCTION is not defined on all compilers,
//comment out if needed
//__func__ is supported in most compilers (C99)
//__FUNCTION__ is supported in all compilers (C++)
#define __func__ __PRETTY_FUNCTION__
inline void ErrorWarning(const char* aFunctionName,
           int aLine, const char *aFile,
           const char* aMessage, bool aAbort)
{
  if(aAbort){
    std::cerr<<"****\nError in "
             <<aFunctionName<<" file: "<<aFile<<", line "<<aLine<<": "
             <<aMessage
             <<"\n****\n\n";
    //cout<<"exiting after returning error..."<<endl;
    exit(1);
  }
  else{
    std::cout<<"****\nWarning in "
             <<aFunctionName<<" file: "<<aFile<<", line "<<aLine<<": "
             <<aMessage
             <<"\n****\n\n";
  }
}

#define ERROR(MESG_)\
  ErrorWarning(__func__,__LINE__,__FILE__,MESG_,true);
#define WARN(MESG_)\
  ErrorWarning(__func__,__LINE__,__FILE__,MESG_,false);

////////////////


typedef long unsigned int counter_t;
typedef float number_t;
typedef std::vector<number_t> TR1; //1d array (rank 1 tensor, but not strictly a tensor)
typedef std::vector<std::vector<number_t> > TR2;//2d array (rank 2 tensor, but not strictly a tensor)
typedef std::vector<std::vector<std::vector<number_t> > > TR3;//3d array (rank 3 tensor, but not strictly a tensor)
typedef std::vector<std::vector<std::vector<std::vector<number_t> > > >TR4;//4d array (rank 4 tensor, but not strictly a tensor)


struct IndexRange {
  counter_t first;
  counter_t last;
};

inline std::ostream & operator<<(std::ostream & aOS, IndexRange &aR) {
  aOS <<aR.first<<" "<< aR.last;
  return aOS;
}

inline std::istream & operator>>(std::istream & aIS, IndexRange &aR) {
  aIS >> aR.first >> aR.last;
  return aIS;
}

struct ConvoPars{
  counter_t span;
  counter_t overlap;
};

inline std::ostream & operator<<(std::ostream & aOS, ConvoPars &aR) {
  aOS <<aR.span<<" "<< aR.overlap;
  return aOS;
}

inline std::istream & operator>>(std::istream & aIS, ConvoPars &aR) {
  aIS >> aR.span >> aR.overlap;
  return aIS;
}
////////////////
//!!New layer types should be added here, and in NLayer::get_layer_type_str() and NLayer::BuildNewLayer()

enum LayerType{kUninitializedLayer, kInputLayer, kFullConnectedLayer,kOutputLayer,
               kLocalReceptiveFieldLayer, kConvolutionLayer, kPoolingLayer,
               kUninitializedLayerGroup, kVerticalLayerGroup, kAggregatingLayerGroup,
               kConvoPoolVLayerGroup};

enum ActivationType{kUninitializedActivation,kLogisticActivation,kReLUActivation,kSoftMaxActivation};
enum CostFunctionType{kUninitializedCost, kQuadraticCost, kXCorrelCost};



inline std::istream & operator>>(std::istream & aIS, ActivationType & aActivationType) {
  //for streaming an enum
  unsigned int nt = 0;
  aIS >> nt;
  aActivationType = static_cast<ActivationType>(nt);
  return aIS;
}

inline std::istream & operator>>(std::istream & aIS, CostFunctionType & aCostFunctionType) {
//for streaming an enum
  unsigned int nt = 0;
  aIS >> nt;
  aCostFunctionType = static_cast<CostFunctionType>(nt);
  return aIS;
}

//for being able to read a LayerType from a stream
inline std::istream & operator>>(std::istream & argIS, LayerType & aLayerType) {
  //for streaming an enum
  unsigned int nt = 0;
  argIS >> nt;
  aLayerType = static_cast<LayerType>(nt);
  return argIS;
}








#endif // __LAYERDNN_TYPES_H__
