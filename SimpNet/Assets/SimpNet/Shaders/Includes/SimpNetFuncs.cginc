#ifndef _SIMPNETFUNCS
#define _SIMPNETFUNCS

#include "Includes/SimpNetLayout.cginc"

inline float afn(float x)
{
    // ELU
    return x >= 0.0 ? x : alpha * (exp(x) - 1.0);
}

inline float dfn(float x)
{
    // ELU
    return x >= 0.0 ? 1.0 : alpha * exp(x);
}

// Keep a weighted history of gradients for RMSProp
inline float momentum(float grad, float vd_m)
{
    return (rho * vd_m) + ((1.0 - rho) * grad * grad);
}

float testImage(uint i, uint j, uint k) {
    return (i / 64.0 * ((64 - j) / 64.0)) + 0.2 * k;
}

float getWL1(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 3 + i.x * 9;
    pos.y = i.w;
    return tex.Load(uint3(txL1Area.xy + txWL1.xy + pos, 0)).x;
}

float getBL1(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txL1Area.xy + txBL1.xy + pos, 0)).x;
}

float getL1s(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 32;
    pos.y = i.z;
    return tex.Load(uint3(txL1Area.xy + txL1s.xy + pos, 0)).x;
}

float getL1a(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 32;
    pos.y = i.z;
    return tex.Load(uint3(txL1Area.xy + txL1a.xy + pos, 0)).x;
}

float getL1Max(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 16;
    pos.y = i.z;
    return tex.Load(uint3(txL1Area.xy + txL1Max.xy + pos, 0)).x;
}

float getL1iMax(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 16;
    pos.y = i.z;
    return tex.Load(uint3(txL1Area.xy + txL1iMax.xy + pos, 0)).x;
}

float getWL2(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 32 + i.x * 96;
    pos.y = i.w;
    return tex.Load(uint3(txL2Area.xy + txWL2.xy + pos, 0)).x;
}

float getBL2(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txL2Area.xy + txBL2.xy + pos, 0)).x;
}

float getL2s(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 14;
    pos.y = i.z;
    return tex.Load(uint3(txL2Area.xy + txL2s.xy + pos, 0)).x;
}

float getL2a(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 14;
    pos.y = i.z;
    return tex.Load(uint3(txL2Area.xy + txL2a.xy + pos, 0)).x;
}

float getL2Max(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 7;
    pos.y = i.z;
    return tex.Load(uint3(txL2Area.xy + txL2Max.xy + pos, 0)).x;
}

float getL2iMax(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 7;
    pos.y = i.z;
    return tex.Load(uint3(txL2Area.xy + txL2iMax.xy + pos, 0)).x;
}


float getWL3(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 64 + i.x * 192;
    pos.y = i.w;
    return tex.Load(uint3(txL3Area.xy + txWL3.xy + pos, 0)).x;
}

float getBL3(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txL3Area.xy + txBL3.xy + pos, 0)).x;
}

float getL3s(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 3;
    pos.y = i.z;
    return tex.Load(uint3(txL3Area.xy + txL3s.xy + pos, 0)).x;
}

float getL3a(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 3;
    pos.y = i.z;
    return tex.Load(uint3(txL3Area.xy + txL3a.xy + pos, 0)).x;
}

float getL3Max(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txL3Area.xy + txL3Max.xy + pos, 0)).x;
}

float getL3iMax(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txL3Area.xy + txL3iMax.xy + pos, 0)).x;
}

float getWFC1(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txFC1Area.xy + txWFC1.xy + pos, 0)).x;
}

float getBFC1(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC1Area.xy + txBFC1.xy + pos, 0)).x;
}

float getFC1s(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC1Area.xy + txFC1s.xy + pos, 0)).x;
}

float getFC1a(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC1Area.xy + txFC1a.xy + pos, 0)).x;
}

float getWFC2(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txFC2Area.xy + txWFC2.xy + pos, 0)).x;
}

float getBFC2(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC2Area.xy + txBFC2.xy + pos, 0)).x;
}

float getFC2s(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC2Area.xy + txFC2s.xy + pos, 0)).x;
}

float getFC2a(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC2Area.xy + txFC2a.xy + pos, 0)).x;
}

float getWFC3(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txFC3Area.xy + txWFC3.xy + pos, 0)).x;
}

float getBFC3(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC3Area.xy + txBFC3.xy + pos, 0)).x;
}

float getFC3s(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC3Area.xy + txFC3s.xy + pos, 0)).x;
}

float getFC3o(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txFC3Area.xy + txFC3o.xy + pos, 0)).x;
}

float getDBFC3(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC3.xy + pos, 0)).x;
}

float getDBFC3_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC3_m.xy + pos, 0)).x;
}

float getDWFC3(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC3.xy + pos, 0)).x;
}

float getDWFC3_m(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC3_m.xy + pos, 0)).x;
}

float getDBFC2(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC2.xy + pos, 0)).x;
}

float getDBFC2_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC2_m.xy + pos, 0)).x;
}

float getDWFC2(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC2.xy + pos, 0)).x;
}

float getDWFC2_m(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC2_m.xy + pos, 0)).x;
}

float getDBFC1(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC1.xy + pos, 0)).x;
}

float getDBFC1_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB4Area.xy + txDBFC1_m.xy + pos, 0)).x;
}

float getDWFC1(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC1.xy + pos, 0)).x;
}

float getDWFC1_m(Texture2D<float> tex, uint2 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y;
    return tex.Load(uint3(txB4Area.xy + txDWFC1_m.xy + pos, 0)).x;
}

float getEMax3(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB3Area.xy + txEMax3.xy + pos, 0)).x;
}

float getEL3(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 3;
    pos.y = i.z;
    return tex.Load(uint3(txB3Area.xy + txEL3.xy + pos, 0)).x;
}

float getDbL3(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB3Area.xy + txDbL3.xy + pos, 0)).x;
}

float getDbL3_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB3Area.xy + txDbL3_m.xy + pos, 0)).x;
}

float getDiL3(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 5;
    pos.y = i.z;
    return tex.Load(uint3(txB3Area.xy + txDiL3.xy + pos, 0)).x;
}

float getDwL3(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 64 + i.x * 192;
    pos.y = i.w;
    return tex.Load(uint3(txB3Area.xy + txDwL3.xy + pos, 0)).x;
}

float getDwL3_m(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 64 + i.x * 192;
    pos.y = i.w;
    return tex.Load(uint3(txB3Area.xy + txDwL3_m.xy + pos, 0)).x;
}

float getPadL3(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 9;
    pos.y = i.z;
    return tex.Load(uint3(txB2Area.xy + txPadL3.xy + pos, 0)).x;
}

float getEL2Max(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 7;
    pos.y = i.z;
    return tex.Load(uint3(txB2Area.xy + txEL2Max.xy + pos, 0)).x;
}

float getDbL2(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB2Area.xy + txDbL2.xy + pos, 0)).x;
}

float getDbL2_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB2Area.xy + txDbL2_m.xy + pos, 0)).x;
}

float getEL2(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 14;
    pos.y = i.z;
    return tex.Load(uint3(txB2Area.xy + txEL2.xy + pos, 0)).x;
}

float getDwL2(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 32 + i.x * 96;
    pos.y = i.w;
    return tex.Load(uint3(txB2Area.xy + txDwL2.xy + pos, 0)).x;
}

float getDwL2_m(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 32 + i.x * 96;
    pos.y = i.w;
    return tex.Load(uint3(txB2Area.xy + txDwL2_m.xy + pos, 0)).x;
}

float getPadL2(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 18;
    pos.y = i.z;
    return tex.Load(uint3(txB1Area.xy + txPadL2.xy + pos, 0)).x;
}

float getEL1Max(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 16;
    pos.y = i.z;
    return tex.Load(uint3(txB1Area.xy + txEL1Max.xy + pos, 0)).x;
}

float getEL1(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + i.x * 32;
    pos.y = i.z;
    return tex.Load(uint3(txB1Area.xy + txEL1.xy + pos, 0)).x;
}

float getDbL1(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB1Area.xy + txDbL1.xy + pos, 0)).x;
}

float getDbL1_m(Texture2D<float> tex, uint i)
{
    int2 pos;
    pos.x = 0;
    pos.y = i;
    return tex.Load(uint3(txB1Area.xy + txDbL1_m.xy + pos, 0)).x;
}

float getDiL1(Texture2D<float> tex, uint3 i)
{
    int2 pos;
    pos.x = i.y + (i.z % 8) * 63;
    pos.y = i.x + (i.z / 8) * 63;
    return tex.Load(uint3(txB1Area.xy + txDiL1.xy + pos, 0)).x;
}

float getDwL1(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 3 + i.x * 9;
    pos.y = i.w;
    return tex.Load(uint3(txB1Area.xy + txDwL1.xy + pos, 0)).x;
}

float getDwL1_m(Texture2D<float> tex, uint4 i)
{
    int2 pos;
    pos.x = i.z + i.y * 3 + i.x * 9;
    pos.y = i.w;
    return tex.Load(uint3(txB1Area.xy + txDwL1_m.xy + pos, 0)).x;
}
#endif