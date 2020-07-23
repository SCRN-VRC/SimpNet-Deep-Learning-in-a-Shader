Shader "SimpNet/SimpNetB4"
{
    Properties
    {
        _BackProp3 ("Backprop 3", 2D) = "black" {}
        _Layer2 ("Layer 2", 2D) = "black" {}
        _Layer1 ("Layer 1", 2D) = "black" {}
        _CamIn ("Camera Input", 2D) = "black" {}
        _FrameBuffer ("Backprop 4 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 4"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _BackProp3;
            Texture2D<float3> _Layer2;
            Texture2D<float3> _Layer1;
            Texture2D<float3> _CamIn;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                int ct = int(_FrameBuffer.Load(int3(_FrameBuffer_TexelSize.zw - 1, 0)).x);

                [branch]
                if (ct == 0 && insideArea(txPConv2Area, px))
                {
                    px -= txPConv2Area.xy;
                    int i = px.y % 18;
                    int j = px.x % 18;
                    int k = px.x / 18 + (px.y / 18) * 8;

                    col.r = i < 2 || j < 2 || i > 15 || j > 15 ? 0.0 : getEConv2(_BackProp3, int3(j - 2, i - 2, k));
                }
                else if (ct == 1 && insideArea(txEMax1Area, px))
                {
                    px -= txEMax1Area.xy;
                    int i = px.y % 16;
                    int j = px.x % 16;
                    int k = px.x / 16 + (px.y / 16) * 4;

                    float sum = 0.0;
                    for (int l = 0; l < 64; l++) {
                        sum += getPConv2(_FrameBuffer, int3(j + 0, i + 0, l)) * getKern2(_Layer2, int4(2, 2, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 1, i + 0, l)) * getKern2(_Layer2, int4(2, 1, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 2, i + 0, l)) * getKern2(_Layer2, int4(2, 0, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 0, i + 1, l)) * getKern2(_Layer2, int4(1, 2, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 1, i + 1, l)) * getKern2(_Layer2, int4(1, 1, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 2, i + 1, l)) * getKern2(_Layer2, int4(1, 0, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 0, i + 2, l)) * getKern2(_Layer2, int4(0, 2, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 1, i + 2, l)) * getKern2(_Layer2, int4(0, 1, k, l));
                        sum += getPConv2(_FrameBuffer, int3(j + 2, i + 2, l)) * getKern2(_Layer2, int4(0, 0, k, l));
                    }
                    col.r = sum;
                }
                else if (ct == 2 && insideArea(txDB1Area, px))
                {
                    px -= txDB1Area.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int j = 0; j < 16; j++) {
                        for (int k = 0; k < 16; k++) {
                            sum += getEMax1(_FrameBuffer, int3(j, k, i));
                        }
                    }
                    col.r = sum;
                }
                else if (ct == 3 && insideArea(txEConv1Area, px))
                {
                    px -= txEConv1Area;
                    int i = px.y % 32;
                    int j = px.x % 32;
                    int k = px.x / 32 + (px.y / 32) * 4;
                    int i0 = i / 2, j0 = j / 2;
                    
                    col.r = abs(getIMax1(_Layer1, int3(j0, i0, k)) - float(i * 32 + j)) < eps ?
                        getEMax1(_FrameBuffer, int3(j0, i0, k)) : 0.0;
                }
                else if (ct == 4 && insideArea(txDiConv1Area, px))
                {
                    px -= txDiConv1Area.xy;
                    int i = px.y % 63;
                    int j = px.x % 63;
                    int k = px.x / 63 + (px.y / 63) * 4;
                    int i0 = i / 2, j0 = j / 2;

                    col.r = ((i % 2 == 1) || (j % 2 == 1)) ? 0.0 : getEConv1(_FrameBuffer, int3(j0, i0, k));
                }
                else if (ct == 5 && insideArea(txDKern1Area, px))
                {
                    px -= txDKern1Area.xy;
                    int i = px.y % 3;
                    int j = px.x % 3;
                    int k = (px.y / 3) % 3;
                    int l = px.x / 3 + (px.y / 9) * 8;
                    
                    float sum = 0.0;
                    for (int x = 0; x < 63; x++) {
                        for (int y = 0; y < 63; y++) {
                            int l1x = x + i;
                            int l1y = y + j;
                            sum += testImage(l1x, l1y, k) * getDiConv1(_FrameBuffer, int3(y, x, l));
                        }
                    }
                    col.r = sum;
                }

                ct = min(ct + 1, 6);
                StoreValue(_FrameBuffer_TexelSize.zw - 1, ct, col.r, px);
                return col;
            }
            ENDCG
        }
    }
}