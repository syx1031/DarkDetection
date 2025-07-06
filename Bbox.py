from pydantic import BaseModel, Field, confloat


class Location(BaseModel):
    x: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s left edge to the total video width.")
    y: confloat(ge=0.0, le=1.0) = Field(...,
                                        description="The ratio of the UI element’s lower edge to the total video height.")
    width: confloat(ge=0.0, le=1.0) = Field(...,
                                            description="The ratio of the UI element’s width to the total video width.")
    height: confloat(ge=0.0, le=1.0) = Field(...,
                                             description="The ratio of the UI element’s height to the total video height.")

Bbox_Description = ''''x' represents the ratio of the UI element’s left edge to the total video width. 'y' represents the ratio of the UI element’s lower edge to the total video height. 'width' represents the ratio of the UI element’s width to the total video width. 'height' represents the ratio of the UI element’s height to the total video height.'''
