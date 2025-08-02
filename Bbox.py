from pydantic import BaseModel, Field, confloat


class Location(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="Normalized horizontal coordinate of the bounding box's lower-left corner. 0.0 represents the far left of the video frame.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="Normalized vertical coordinate of the bounding box's lower-left corner. 0.0 represents the very bottom of the video frame.")
    width: confloat(ge=0.0, le=1.0) = Field(...,
                                            description="Normalized width of the bounding box, as a fraction of the video frame's total width.")
    height: confloat(ge=0.0, le=1.0) = Field(...,
                                             description="Normalized height of the bounding box, as a fraction of the video's total height.")

# Bbox_Description = ''''x' represents the ratio of the UI element’s left edge to the total video width. 'y' represents the ratio of the UI element’s lower edge to the total video height. 'width' represents the ratio of the UI element’s width to the total video width. 'height' represents the ratio of the UI element’s height to the total video height.'''

Bbox_Description = '''
The location is described using a bounding box with the following schema:
- Coordinate System: The origin (0, 0) is the bottom-left corner of the video.
- x: The normalized horizontal coordinate of the box's left edge, from 0.0 to 1.0.
- y: The normalized vertical coordinate of the box's bottom edge, from 0.0 to 1.0.
- width: The normalized width of the box, from 0.0 to 1.0.
- height: The normalized height of the box, from 0.0 to 1.0.
'''
