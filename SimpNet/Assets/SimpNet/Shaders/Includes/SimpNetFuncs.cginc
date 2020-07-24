#ifndef _SIMPNETFUNCS
#define _SIMPNETFUNCS

#include "Includes/SimpNetLayout.cginc"

inline float actFn(float x)
{
    // ELU
    return x >= 0.0f ? x : (0.15f * (exp(x) - 1.0f));
}

inline float dactFn(float x)
{
    // ELU
    return x >= 0.0f ? 1.0f : exp(x) * 0.15f;
}

float testImage(int i, int j, int k) {
    return (i / 65.0 * ((65 - j) / 65.0)) + 0.2 * k;
}

float getKern1(Texture2D<float> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + ((i.w % 8) * 3);
    pos.y = i.y + i.z * 3 + ((i.w / 8) * 9);
    return tex.Load(int3(txL1.xy + txKern1Area.xy + pos, 0)).x;
}

float getConv1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 32;
    pos.y = i.y + (i.z / 4) * 32;
    return tex.Load(int3(txL1.xy + txConv1Area.xy + pos, 0)).x;
}

float getMax1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 2) * 16;
    pos.y = i.y + (i.z / 2) * 16;
    return tex.Load(int3(txL1.xy + txMax1Area.xy + pos, 0)).x;
}

float getKern2(Texture2D<float> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + i.z * 3;
    pos.y = i.y + i.w * 3;
    return tex.Load(int3(txL2.xy + txKern2Area.xy + pos, 0)).x;
}

float getConv2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 14;
    pos.y = i.y + (i.z / 8) * 14;
    return tex.Load(int3(txL2.xy + txConv2Area.xy + pos, 0)).x;
}

float getMax2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txL2.xy + txMax2Area.xy + pos, 0)).x;
}

float getKern3(Texture2D<float> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + i.z * 3;
    pos.y = i.y + i.w * 3;
    return tex.Load(int3(txL3.xy + txKern3Area.xy + pos, 0)).x;
}

float getConv3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 4;
    return tex.Load(int3(txL3.xy + txConv3Area.xy + pos, 0)).x;
}

float getMax3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txL3.xy + txMax3Area.xy + pos, 0)).x;
}

float getW1(Texture2D<float> tex, int4 i)
{
    int2 pos;
    pos.y = i.x + i.z * 2;
    pos.x = i.y + i.w * 2;
    return tex.Load(int3(txL4.xy + txW1Area.xy + pos, 0)).x;
}

float getW3(Texture2D<float> tex, int2 i)
{
    int2 pos;
    pos.x = i.x + (i.y / 64) * 12;
    pos.y = i.y % 64;
    return tex.Load(int3(txL6.xy + txW3Area.xy + pos, 0)).x;
}

float getEMax3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txB2.xy + txEMax3Area.xy + pos, 0)).x;
}

float getIMax3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txL3.xy + txiMax3Area.xy + pos, 0)).x;
}

float getEConv3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 4;
    return tex.Load(int3(txB2.xy + txEConv3Area.xy + pos, 0)).x;
}

float getDiConv3(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txB2.xy + txDiConv3Area.xy + pos, 0)).x;
}

float getEMax2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txB3.xy + txEMax2Area.xy + pos, 0)).x;
}

float getIMax2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txL2.xy + txiMax2Area.xy + pos, 0)).x;
}

float getEConv2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 14;
    pos.y = i.y + (i.z / 8) * 14;
    return tex.Load(int3(txB3.xy + txEConv2Area.xy + pos, 0)).x;
}

float getPConv2(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 18;
    pos.y = i.y + (i.z / 8) * 18;
    return tex.Load(int3(txB4.xy + txPConv2Area.xy + pos, 0)).x;
}

float getEMax1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 16;
    pos.y = i.y + (i.z / 4) * 16;
    return tex.Load(int3(txB4.xy + txEMax1Area.xy + pos, 0)).x;
}

float getIMax1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 2) * 16;
    pos.y = i.y + (i.z / 2) * 16;
    return tex.Load(int3(txL1.xy + txiMax1Area.xy + pos, 0)).x;
}

float getEConv1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 32;
    pos.y = i.y + (i.z / 4) * 32;
    return tex.Load(int3(txB4.xy + txEConv1Area.xy + pos, 0)).x;
}

float getDiConv1(Texture2D<float> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 63;
    pos.y = i.y + (i.z / 4) * 63;
    return tex.Load(int3(txB4.xy + txDiConv1Area.xy + pos, 0)).x;
}

#endif