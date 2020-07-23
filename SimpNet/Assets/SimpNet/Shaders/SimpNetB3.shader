Shader "SimpNet/SimpNetB3"
{
    Properties
    {
        _BackProp2 ("Backprop 2", 2D) = "black" {}
        _Layer3 ("Layer 3", 2D) = "black" {}
        _Layer2 ("Layer 2", 2D) = "black" {}
        _Layer1 ("Layer 1", 2D) = "black" {}
        _FrameBuffer ("Backprop 3 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 3"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            //RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _BackProp2;
            Texture2D<float3> _Layer3;
            Texture2D<float3> _Layer2;
            Texture2D<float3> _Layer1;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                int ct = int(_FrameBuffer.Load(int3(_FrameBuffer_TexelSize.zw - 1, 0)).x);

                [branch]
                if (ct == 0 && insideArea(txEMax2Area, px))
                {
                    px -= txEMax2Area.xy;
                    int i = px.y % 7;
                    int j = px.x % 7;
                    int k = px.x / 7 + (px.y / 7) * 8;
                    int i0 = i, i1 = i - 1, i2 = i + 1;
                    int j0 = j, j1 = j - 1, j2 = j + 1;
                    // Padding
                    bool b0 = (i1 < 0 || j1 < 0), b1 = (i1 < 0), b2 = (i1 < 0 || j2 > 6), b3 = (j1 < 0);
                    bool b4 = (j2 > 6), b5 = (i2 > 6 || j1 < 0), b6 = (i2 > 6), b7 = (i2 > 6 || j2 > 6);

                    float sum = 0.0;
                    for (int l = 0; l < 128; l++) {
                        sum += (b0 ? 0.0 : getDiConv3(_BackProp2, int3(j1, i1, l)) * getKern3(_Layer3, int4(2, 2, k, l)));
                        sum += (b1 ? 0.0 : getDiConv3(_BackProp2, int3(j0, i1, l)) * getKern3(_Layer3, int4(2, 1, k, l)));
                        sum += (b2 ? 0.0 : getDiConv3(_BackProp2, int3(j2, i1, l)) * getKern3(_Layer3, int4(2, 0, k, l)));
                        sum += (b3 ? 0.0 : getDiConv3(_BackProp2, int3(j1, i0, l)) * getKern3(_Layer3, int4(1, 2, k, l)));
                        sum += getDiConv3(_BackProp2, int3(j0, i0, l)) * getKern3(_Layer3, int4(1, 1, k, l));
                        sum += (b4 ? 0.0 : getDiConv3(_BackProp2, int3(j2, i0, l)) * getKern3(_Layer3, int4(1, 0, k, l)));
                        sum += (b5 ? 0.0 : getDiConv3(_BackProp2, int3(j1, i2, l)) * getKern3(_Layer3, int4(0, 2, k, l)));
                        sum += (b6 ? 0.0 : getDiConv3(_BackProp2, int3(j0, i2, l)) * getKern3(_Layer3, int4(0, 1, k, l)));
                        sum += (b7 ? 0.0 : getDiConv3(_BackProp2, int3(j2, i2, l)) * getKern3(_Layer3, int4(0, 0, k, l)));
                    }
                    col.r = sum;
                }
                else if (ct == 1 && insideArea(txDB2Area, px))
                {
                    px -= txDB2Area.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int j = 0; j < 7; j++) {
                        for (int k = 0; k < 7; k++) {
                            sum += getEMax2(_FrameBuffer, int3(j, k, i));
                        }
                    }
                    col.r = sum;
                }
                else if (ct == 2 && insideArea(txEConv2Area, px))
                {
                    px -= txEConv2Area.xy;
                    int i = px.y % 14;
                    int j = px.x % 14;
                    int k = px.x / 14 + (px.y / 14) * 8;
                    int i0 = i / 2, j0 = j / 2;

                    col.r = abs(getIMax2(_Layer2, int3(j0, i0, k)) - float(i * 14 + j)) < eps ?
                        getEMax2(_FrameBuffer, int3(j0, i0, k)) : 0.0;
                }
                else if (ct == 3 && insideArea(txDKern2Area, px))
                {
                    px -= txDKern2Area.xy;
                    int i = px.y % 3;
                    int j = px.x % 3;
                    int k = px.x / 3;
                    int l = px.y / 3;

                    float sum = 0.0;
                    for (int x = 0; x < 14; x++) {
                        for (int y = 0; y < 14; y++) {
                            int l1x = x + i;
                            int l1y = y + j;
                            sum += getMax1(_Layer1, int3(l1y, l1x, k)) * getEConv2(_FrameBuffer, int3(y, x, l));
                        }
                    }
                    col.r = sum;
                }

                ct = min(ct + 1, 4);
                StoreValue(_FrameBuffer_TexelSize.zw - 1, ct, col.r, px);
                return col;
            }
            ENDCG
        }
    }
}