from pydantic import BaseModel, Field, confloat


class PointLocation(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s center x-coordinate to the total video width.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s center y-coordinate to the total video height.")


PointLocation_Description = ''''x' represents the ratio of the UI element’s center x-coordinate to the total video width, and 'y' represents the ratio of the UI element’s center y-coordinate to the total video height.'''
