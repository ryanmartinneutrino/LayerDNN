#include "CPVLayerGroup.h"
CPVLayerGroup::CPVLayerGroup()
{
  fLayerType=kConvoPoolVLayerGroup;
}

CPVLayerGroup::CPVLayerGroup(counter_t anSpan, counter_t anPool, counter_t anInput, std::string aLayerID):
                             VLayerGroup(aLayerID)
{
  fLayerType=kConvoPoolVLayerGroup;
  fConvoLayer= new CNLayer(anSpan,0,anInput);
  add_layer(fConvoLayer);
  fPoolLayer= new PNLayer(anPool,fConvoLayer->get_noutput());
  add_layer(fPoolLayer);
}

CPVLayerGroup::~CPVLayerGroup()
{
  //dtor
}
