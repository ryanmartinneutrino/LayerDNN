#ifndef __LAYERREGISTER_H__
#define __LAYERREGISTER_H__

#include "NLayer.h"

class LayerRegister
{ //Class to create and keep track of layers. All derived classes should be "registered" here
  public:
    LayerRegister();
    virtual ~LayerRegister();

    static NLayer* BuildNewLayer(LayerType aType);
    static std::string LayerType2String(LayerType aType);


  protected:


  private:
};

#endif // __LAYERREGISTER_H__
