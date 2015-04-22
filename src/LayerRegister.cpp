#include "LayerRegister.h"
#include "NLayer.h"
#include "ONLayer.h"
#include "FNLayer.h"
#include "INLayer.h"
#include "LNLayer.h"
#include "CNLayer.h"
#include "PNLayer.h"
#include "LayerGroup.h"
#include "VLayerGroup.h"
#include "ALayerGroup.h"
#include "CPVLayerGroup.h"


LayerRegister::LayerRegister()
{

}

LayerRegister::~LayerRegister()
{

}

std::string LayerRegister::LayerType2String(LayerType aType){
  switch(aType){
    case kUninitializedLayer:
      return "Uninitialized";
    case kInputLayer:
      return "Input";
    case kFullConnectedLayer:
      return "FullyConnected";
    case kOutputLayer:
      return "Output";
    case kLocalReceptiveFieldLayer:
      return "LocalReceptiveField";
    case kConvolutionLayer:
      return "Convolution";
    case kPoolingLayer:
      return "Pooling";
    case kUninitializedLayerGroup:
      return "UninitializedLayerGroup";
    case kVerticalLayerGroup:
      return "VerticalLayerGroup";
    case kAggregatingLayerGroup:
      return "AggregatingLayerGroup";
    case kConvoPoolVLayerGroup:
      return "ConvoPoolingVLayerGroup";
    //case kHorizontalLayerGroup:
      //return "Horizontal Layer Group";
    default:
      return "Unknown";
  }
}

NLayer* LayerRegister::BuildNewLayer(LayerType aType){
  switch(aType){
    case kUninitializedLayer:
      return NULL;
    case kInputLayer:
      return new INLayer();
    case kFullConnectedLayer:
      return new FNLayer();
    case kOutputLayer:
      return new ONLayer();
    case kLocalReceptiveFieldLayer:
      return new LNLayer();
    case kConvolutionLayer:
      return new CNLayer();
    case kPoolingLayer:
      return new PNLayer();
    case kAggregatingLayerGroup:
      return new ALayerGroup();
    case kUninitializedLayerGroup:
      return NULL;
    case kVerticalLayerGroup:
      return new VLayerGroup();
    case kConvoPoolVLayerGroup:
      return new CPVLayerGroup();
    //case kHorizontalLayerGroup:
      //return new HLayerGroup();
    default:
      return NULL;
  }
}

