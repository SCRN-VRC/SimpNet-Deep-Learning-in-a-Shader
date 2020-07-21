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

#endif