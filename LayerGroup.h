#ifndef __LAYERGROUP_H__
#define __LAYERGROUP_H__

#include "NLayer.h"
#include "LayerData.h"

#include <vector>

class LayerGroup: public NLayer
{ //This is an abstract class to hold a group of layers
  public:
    LayerGroup(std::string aLayerID="noid");
    virtual ~LayerGroup();

    //inherited functions that are applied all layers in the group
    virtual void print(counter_t aDataIndex=0);//call print() in all layers
    virtual void update_weights();//call update_weights in all layers
    void multiply_global_learning_rate(number_t aFact=1.0);
    void add_to_global_learning_rate(number_t a=0.0);

    //Derived class must implement these:
    virtual void initialize(bool aForce=false)=0;
    virtual void initialize_layer_data()=0;
    virtual void backprop_pass(const TR2 &aY=TR2(0), counter_t aStart=0, counter_t anData=0)=0;
    virtual void forward_pass(const TR2 &aX=TR2(0), counter_t aStart=0, counter_t anData=0)=0;
/*
    void add_layer(NLayer* aLayer){
      fLayer.push_back(aLayer);
      fnLayer=fLayer.size();
      if(fnLayer==1)fFirstLayer=aLayer;
      fLastLayer=aLayer;
    }
*/
    NLayer* get_last_layer(){return fLastLayer;}
    NLayer* get_layer(counter_t aIndex=0){return fLayer[aIndex];}
    LayerData* get_layer_data_ptr(counter_t aIndex=0){return fLayer[aIndex]->get_layer_data_ptr();}

  protected:
    NLayer* fFirstLayer;
    NLayer* fLastLayer;

  private:
};

#endif // __LAYERGROUP_H__
