#ifndef _SIMPNETFUNCS
#define _SIMPNETFUNCS

#include "Includes/SimpNetLayout.cginc"

#define eps 0.00001
#define lr 0.2
#define lrb 0.1

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

float getKern1(Texture2D<float3> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + ((i.w % 8) * 3);
    pos.y = i.y + i.z * 3 + ((i.w / 8) * 9);
    return tex.Load(int3(txKern1Area.xy + pos, 0)).x;
}

float getConv1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 32;
    pos.y = i.y + (i.z / 4) * 32;
    return tex.Load(int3(txConv1Area.xy + pos, 0)).x;
}

float getMax1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 2) * 16;
    pos.y = i.y + (i.z / 2) * 16;
    return tex.Load(int3(txMax1Area.xy + pos, 0)).x;
}

float getKern2(Texture2D<float3> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + i.z * 3;
    pos.y = i.y + i.w * 3;
    return tex.Load(int3(txKern2Area.xy + pos, 0)).x;
}

float getConv2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 14;
    pos.y = i.y + (i.z / 8) * 14;
    return tex.Load(int3(txConv2Area.xy + pos, 0)).x;
}

float getMax2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txMax2Area.xy + pos, 0)).x;
}

float getKern3(Texture2D<float3> tex, int4 i)
{
    int2 pos;
    pos.x = i.x + i.z * 3;
    pos.y = i.y + i.w * 3;
    return tex.Load(int3(txKern3Area.xy + pos, 0)).x;
}

float getConv3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 4;
    return tex.Load(int3(txConv3Area.xy + pos, 0)).x;
}

float getMax3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txMax3Area.xy + pos, 0)).x;
}

float getW1(Texture2D<float3> tex, int4 i)
{
    int2 pos;
    pos.y = i.x + i.z * 2;
    pos.x = i.y + i.w * 2;
    return tex.Load(int3(txW1Area.xy + pos, 0)).x;
}

float getW3(Texture2D<float3> tex, int2 i)
{
    int2 pos;
    pos.x = i.x + (i.y / 64) * 12;
    pos.y = i.y % 64;
    return tex.Load(int3(txW3Area.xy + pos, 0)).x;
}

float getEMax3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txEMax3Area.xy + pos, 0)).x;
}

float getIMax3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 2;
    return tex.Load(int3(txiMax3Area.xy + pos, 0)).x;
}

float getEConv3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x;
    pos.y = i.y + i.z * 4;
    return tex.Load(int3(txEConv3Area.xy + pos, 0)).x;
}

float getDiConv3(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txDiConv3Area.xy + pos, 0)).x;
}

float getEMax2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txEMax2Area.xy + pos, 0)).x;
}

float getIMax2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 7;
    pos.y = i.y + (i.z / 8) * 7;
    return tex.Load(int3(txiMax2Area.xy + pos, 0)).x;
}

float getEConv2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 14;
    pos.y = i.y + (i.z / 8) * 14;
    return tex.Load(int3(txEConv2Area.xy + pos, 0)).x;
}

float getPConv2(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 8) * 18;
    pos.y = i.y + (i.z / 8) * 18;
    return tex.Load(int3(txPConv2Area.xy + pos, 0)).x;
}

float getEMax1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 16;
    pos.y = i.y + (i.z / 4) * 16;
    return tex.Load(int3(txEMax1Area.xy + pos, 0)).x;
}

float getIMax1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 2) * 16;
    pos.y = i.y + (i.z / 2) * 16;
    return tex.Load(int3(txiMax1Area.xy + pos, 0)).x;
}

float getEConv1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 32;
    pos.y = i.y + (i.z / 4) * 32;
    return tex.Load(int3(txEConv1Area.xy + pos, 0)).x;
}

float getDiConv1(Texture2D<float3> tex, int3 i)
{
    int2 pos;
    pos.x = i.x + (i.z % 4) * 63;
    pos.y = i.y + (i.z / 4) * 63;
    return tex.Load(int3(txDiConv1Area.xy + pos, 0)).x;
}

#endif