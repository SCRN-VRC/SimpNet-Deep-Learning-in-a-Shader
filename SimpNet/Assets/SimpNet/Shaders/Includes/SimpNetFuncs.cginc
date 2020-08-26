#ifndef _SIMPNETFUNCS
#define _SIMPNETFUNCS

#include "Includes/SimpNetLayout.cginc"

inline float afn(float x)
{
    // ELU
    return x >= 0.f ? x : alpha * (exp(x) - 1.f);
}

inline float dfn(float x)
{
    // ELU
    return x >= 0.f ? 1.f : alpha * exp(x);
}

float testImage(uint i, uint j, uint k) {
    return (i / 64.0 * ((64 - j) / 64.0)) + 0.2 * k;
}

float getWL1(Texture2D<float> tex, int4 i)
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

#endif