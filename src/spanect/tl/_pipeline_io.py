"""Raw pipeline I/O: compatibility facade delegating to pp + gr."""


def get_input(
        adata,
        data_path,
        data_name,
        cnnType='ResNet50',
        device='cuda:3',
        k=20,
):
    """Delegate to gr.prepare_graph_inputs (which uses pp + graph ops)."""
    from ..gr._pipeline_graph import prepare_graph_inputs
    return prepare_graph_inputs(
        adata=adata,
        data_path=data_path,
        data_name=data_name,
        cnnType=cnnType,
        device=device,
        k=k,
    )


__all__ = ["get_input"]
