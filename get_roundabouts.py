import sys
import os
import lxml.etree
import gdal
import osr


def connect_ways(way_key, way_dict, ways_of_node, ret_ways):
    '''
    helper function to cdetermine which ways belong together
        (a roundabout may consist of multiple ways)
    '''

    if way_key in ret_ways:
        return

    ret_ways.append(way_key)

    for node_key in way_dict[way_key]:
        if node_key in ways_of_node:
            if len(ways_of_node[node_key])>1:
                for way_key2 in ways_of_node[node_key]:
                    connect_ways(way_key2, way_dict, ways_of_node, ret_ways)


def get_roundabouts(inp_filename):
    '''
    returns the min/max coordinates of all roundabouts in an osm file
    '''

    inp_doc   = lxml.etree.parse(inp_filename).getroot()
    node_list = inp_doc.xpath("node")
    way_list  = inp_doc.xpath("way")


    #load all nodes into dict with node_id as key and coordinate tuples as value
    node_dict = dict()
    for n in node_list:
        node_dict[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))


    #load all ways into dict with way_id as key and node_key-list as value
    way_dict = dict()
    for w in way_list:
        way_dict[w.get('id')] = list()
        for n in w.xpath("nd"):
            way_dict[w.get('id')].append(n.get('ref'))


    #for each node, check in which ways it is used and save in dict with
    #   node_id as key and way_id-list as value. This check is necessary as a
    #   roundabout can consist of several ways.
    ways_of_node = dict()
    for node_key in node_dict.keys():

        ways_list=list()

        for way_key in way_dict.keys():
            if node_key in way_dict[way_key]:
                ways_list.append(way_key)
        
        ways_of_node[node_key]=ways_list


    #connect ways together (using information in ways_of_node)
    checked_way_keys=list()
    connected_ways=list()

    for way_key in way_dict:
        if way_key not in checked_way_keys:
            ret_ways=list()
            connect_ways(way_key, way_dict, ways_of_node, ret_ways)
            connected_ways.append(ret_ways)
            checked_way_keys.extend(ret_ways)


    ret_val=list()
    #print connected_ways
    for way_list in connected_ways:
        #init min/max with first point of first way
        if way_list[0] not in way_dict:
            continue

        if way_dict[way_list[0]][0] not in node_dict:
            continue

        init_coords = node_dict[way_dict[way_list[0]][0]]
        max_lat = min_lat = init_coords[0]
        max_lon = min_lon = init_coords[1]

        for way in way_list:
            for node in way_dict[way]:
                
                if node not in node_dict:
                    continue

                coords = node_dict[node]
                max_lat = max(max_lat,coords[0])
                min_lat = min(min_lat,coords[0])
                max_lon = max(max_lon,coords[1])
                min_lon = min(min_lon,coords[1])
        ret_val.append((way_list, [min_lon, min_lat, max_lon, max_lat]))
        #print "way_id:", way_list
        #print min_lon, min_lat, max_lon, max_lat

    return ret_val


def get_geo_corners(img_filename):
    '''
    get image corners in Geographic coordinates
    '''

    geotiff      = gdal.Open(img_filename)
    gt_transform = geotiff.GetGeoTransform()
   
    #get coordinates in coordinate system of image
    x_tl = gt_transform[0]
    y_tl = gt_transform[3]
    x_br = x_tl + geotiff.RasterXSize * gt_transform[1]
    y_br = y_tl + geotiff.RasterYSize * gt_transform[5]


    #transform coordinates to Geographic
    src_projection = geotiff.GetProjection()
    src_reference  = osr.SpatialReference()
    src_reference.ImportFromWkt(src_projection)

    dst_reference = osr.SpatialReference()
    dst_reference.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src_reference, dst_reference)
    x_tl,y_tl,dummy=transform.TransformPoint(x_tl,y_tl,0)
    x_br,y_br,dummy=transform.TransformPoint(x_br,y_br,0)

    return x_tl,y_br,x_br,y_tl

    
def geo_to_img_coords(img_filename, x, y):
    '''
    transform geographic coordinates, to pixel coordinates
    '''

    geotiff      = gdal.Open(img_filename)

    #transform coordinates from Geographic to image geo-coordinates
    dst_projection = geotiff.GetProjection()
    dst_reference = osr.SpatialReference()
    dst_reference.ImportFromWkt(dst_projection)

    src_reference = osr.SpatialReference()
    src_reference.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src_reference, dst_reference)
    ret_x,ret_y,dummy=transform.TransformPoint(x,y,0)

    #transform from image geo-coordinates to pixel coordinates
    gt_transform = geotiff.GetGeoTransform()
    ret_x = (ret_x - gt_transform[0]) / gt_transform[1]
    ret_y = (ret_y - gt_transform[3]) / gt_transform[5]

    return ret_x, ret_y
     

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Usage:"
        print "get_roundabouts.py file.osm img.tif"
        sys.exit(2)

    #get image corners in geographic
    geo_corners = get_geo_corners(sys.argv[2])


    osmconvert_rect = "%f,%f,%f,%f" % geo_corners
    os.system("./osmconvert32 "+sys.argv[1]+" -b="+ osmconvert_rect +" > sub_section.osm")

    roundabouts = get_roundabouts("sub_section.osm")

    for r in roundabouts:
        l_l = geo_to_img_coords(sys.argv[2], r[1][0], r[1][1])
        u_r = geo_to_img_coords(sys.argv[2], r[1][2], r[1][3])
        print "%f,%f,%f,%f" % (l_l[0], l_l[1], u_r[0], u_r[1])

