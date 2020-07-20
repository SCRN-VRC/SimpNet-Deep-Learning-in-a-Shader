Shader "SimpNet/SimpNetL4"
{
    Properties
    {
        _Layer3 ("Layer 3", 2D) = "black" {}
        _L4Gradients ("Layer 4 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 4 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 4"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _Layer3;
            Texture2D<float3> _L4Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer3_TexelSize;
            float4 _L4Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txW1Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txW1BiasArea, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txFC1s, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txFC1a, px))
                {
                    col.r = 0.7;
                }

                return col;
            }
            ENDCG
        }
    }
}