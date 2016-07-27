from django import forms

from django import forms
from models import Category, Blank

class CategoryForm(forms.ModelForm):
    fb1 = forms.CheckboxInput()
    fb2 = forms.CheckboxInput()
    fb3 = forms.CheckboxInput()
    fb4 = forms.CheckboxInput()
    fb5 = forms.CheckboxInput()
    fb6 = forms.CheckboxInput()
   
    class Meta:
       
        model = Category
        fields = ('fb1','fb2','fb3','fb4','fb5','fb6')

class BlankForm(forms.ModelForm):
    fb1 = forms.HiddenInput()

    class Meta:
        # Provide an association between the ModelForm and a model
        model = Blank

