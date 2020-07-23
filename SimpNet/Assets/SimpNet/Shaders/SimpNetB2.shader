Shader "SimpNet/SimpNetB2"
{
    Properties
    {
        _BackProp1 ("Backprop 1", 2D) = "black" {}
        _Layer4 ("Layer 4", 2D) = "black" {}
        _Layer3 ("Layer 3", 2D) = "black" {}
        _Layer2 ("Layer 2", 2D) = "black" {}
        _FrameBuffer ("Backprop 2 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 2"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float> _BackProp1;
            Texture2D<float> _Layer4;
            Texture2D<float> _Layer3;
            Texture2D<float> _Layer2;
            Texture2D<float> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float col = _FrameBuffer.Load(int3(px, 0));

                int ct = int(_FrameBuffer.Load(int3(_FrameBuffer_TexelSize.zw - 1, 0)).x);
                buffer[0].y = ct;

                [branch]
                if (ct == 1 && insideArea(txEMax3Area, px))
                {
                    px -= txEMax3Area.xy;
                    int i = px.y % 2;
                    int j = px.x % 2;
                    int k = px.y / 2;

                    float sum = 0.0;
                    for (int l = 0; l < 128; l++) {
                        sum += _BackProp1.Load(int3(txDBW1Area + int2(0, l), 0)).x *
                            dactFn(_Layer4.Load(int3(txFC1s.xy + int2(0, l), 0)).x) *
                            getW1(_Layer4, int4(i, j, k, l));
                    }
                    col.r = sum;
                }
                else if (ct == 2 && insideArea(txDB3Area, px))
                {
                    px -= txDB3Area.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int j = 0; j < 2; j++) {
                        for (int k = 0; k < 2; k++) {
                            sum += getEMax3(_FrameBuffer, int3(j, k, i));
                        }
                    }
                    col.r = sum;
                }
                else if (ct == 3 && insideArea(txEConv3Area, px))
                {
                    px -= txEConv3Area.xy;
                    int i = px.y % 4;
                    int j = px.x % 4;
                    int k = px.y / 4;
                    int i0 = i / 2, j0 = j / 2;

                    col.r = abs(getIMax3(_Layer3, int3(j0, i0, k)) - float(i * 4 + j)) < eps ?
                        getEMax3(_FrameBuffer, int3(j0, i0, k)) : 0.0;
                }
                else if (ct == 4 && insideArea(txDiConv3Area, px))
                {
                    px -= txDiConv3Area.xy;
                    int i = px.y % 7;
                    int j = px.x % 7;
                    int k = px.x / 7 + (px.y / 7) * 8;
                    int i0 = i / 2, j0 = j / 2;
                    
                    col.r = ((i % 2 == 1) || (j % 2 == 1)) ? 0.0 : getEConv3(_FrameBuffer, int3(j0, i0, k));
                }
                else if (ct == 0 && insideArea(txDKern3Area, px))
                {
                    px -= txDKern3Area.xy;
                    int i = px.y % 3;
                    int j = px.x % 3;
                    int k = px.x / 3;
                    int l = px.y / 3;

                    float sum = 0.0;
                    for (int x = 0; x < 7; x++) {
                        for (int y = 0; y < 7; y++) {
                            int l2x = x + i - 1;
                            int l2y = y + j - 1;
                            // Padding
                            bool b = l2x < 0 || l2y < 0 || l2x > 6 || l2y > 6;
                            sum += b ? 0.0 : getMax2(_Layer2, int3(l2y, l2x, k)) * getDiConv3(_FrameBuffer, int3(y, x, l));
                        }
                    }
                    col.r = sum;
                }

                ct = (ct + 1) % B2_MAX_CT;
                StoreValue(_FrameBuffer_TexelSize.zw - 1, ct, col, px);
                return col;
            }
            ENDCG
        }
    }
}