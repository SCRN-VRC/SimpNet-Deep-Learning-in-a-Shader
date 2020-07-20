Shader "SimpNet/SimpNetB4"
{
    Properties
    {
        _Layer1 ("Backprop 3", 2D) = "black" {}
        _FrameBuffer ("Backprop 4 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 4"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _CamIn;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txPConv2Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txEMax1Area, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txDB1Area, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txEConv1Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txDiConv1Area, px))
                {
                    col.r = 0.6;
                }
                else if (insideArea(txDKern1Area, px))
                {
                    col.r = 0.5;
                }

                return col;
            }
            ENDCG
        }
    }
}