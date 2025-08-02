from pydantic import BaseModel, Field, confloat


class PointLocation(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="Normalized horizontal coordinate of the point (UI element's center). 0.0 represents the far left of the video frame.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="Normalized vertical coordinate of the point (UI element's center). 0.0 represents the very bottom of the video frame.")


PointLocation_Description = '''
The location is described using a single point (representing the UI element's center) with the following schema:
- Coordinate System: The origin (0, 0) is the bottom-left corner of the video.
- x: The normalized horizontal coordinate of the point, from 0.0 (left edge) to 1.0 (right edge).
- y: The normalized vertical coordinate of the point, from 0.0 (bottom edge) to 1.0 (top edge).
'''
