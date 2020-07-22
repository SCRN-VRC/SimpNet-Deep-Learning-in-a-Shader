#ifndef _SIMPNETFUNCS
#define _SIMPNETFUNCS

#include "Includes/SimpNetLayout.cginc"

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

#endif