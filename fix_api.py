import json

with open('/Users/jack/Documents/gitcode/serveless-allinone/workflows/wan2_2_animate_dwpose_base_api.json', 'r') as f:
    api = json.load(f)

with open('/Users/jack/Documents/gitcode/serveless-allinone/Wan2_2Animate_DWPose.json', 'r') as f:
    ui = json.load(f)

# Create lookup from UI nodes
ui_nodes = {str(n['id']): n for n in ui['nodes']}

for node_id, node_data in api.items():
    if node_id not in ui_nodes:
        continue
        
    ui_node = ui_nodes[node_id]
    
    # 1. Fix missing class_type
    if 'class_type' not in node_data:
        node_data['class_type'] = ui_node['type']
        
    # 2. Fix UNKNOWN keys
    if 'inputs' in node_data and 'UNKNOWN' in node_data['inputs']:
        # ComfyUI sometimes assigns UNKNOWN if it can't map the widget.
        # Let's map widgets directly from UI JSON.
        widgets_values = ui_node.get('widgets_values', [])
        
        # We need to find the widget names. In UI json, inputs array might contain widgets.
        # Actually, in ComfyUI, the order in `widgets_values` corresponds to the widgets.
        # But we don't always know their names unless they are explicitly in inputs as {"widget": {"name": "..."}}
        widget_names = []
        if 'inputs' in ui_node:
            for inp in ui_node['inputs']:
                if 'widget' in inp and 'name' in inp['widget']:
                    widget_names.append(inp['widget']['name'])
        
        # Or from properties?
        if 'properties' in ui_node and 'Node name for S&R' in ui_node['properties']:
            node_data['_meta']['title'] = ui_node['properties']['Node name for S&R']
            
        # Delete UNKNOWN
        del node_data['inputs']['UNKNOWN']
        
        # If we have exactly 1 widget value and 1 widget name
        if isinstance(widgets_values, list):
            # It's tricky to map them perfectly without the backend schema, but let's try our best.
            pass

with open('/Users/jack/Documents/gitcode/serveless-allinone/workflows/wan2_2_animate_dwpose_base_api_fixed.json', 'w') as f:
    json.dump(api, f, indent=2)
