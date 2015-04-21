#ifndef __FLAYERDATA_H__
#define __FLAYERDATA_H__

#include "LayerData.h"

class FNLayer;

class FLayerData : public LayerData
{
  public:
    FLayerData(FNLayer* aLayer=NULL);
    virtual ~FLayerData();


  protected:
  private:
};

#endif // __FLAYERDATA_H__
