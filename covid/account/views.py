# Project Name:       COVID-19 Care: Face Mask and Social Distancing Detection using Deep Learning
# Author List:        Nikhil Joshi
# Filename:           views.py
# Functions:          homepage_view(request), login_view(request), logout_view(request), register_view(request), about_us_view(request), contact_us_view(request), account_view(request)
# Global Variables:   NA

from django.shortcuts import render, redirect
from account.forms import *
from django.contrib.auth import logout, login, authenticate
from django.http import HttpResponse


# Function Name:	homepage_view
# Input:		    HTTP request
# Output:		    Returns the template of the homepage
# Logic:		    Return and render the template
# Example Call:		Given by Django framework
def homepage_view(request):
    return render(request, 'account/home.html')


# Function Name:	login_view
# Input:		    HTTP request
# Output:		    User login
# Logic:		    Get the user currently associated, get the email and password and login the user
# Example Call:		Given by Django framework
def login_view(request):

	context = {}

	user = request.user
	if user.is_authenticated: 
		return redirect("homepage")

	if request.POST:
		form = AccountAuthenticationForm(request.POST)
		if form.is_valid():
			email = request.POST['email']
			password = request.POST['password']
			user = authenticate(email=email, password=password)

			if user:
				login(request, user)
				return redirect("homepage")

	else:
		form = AccountAuthenticationForm()

	context['login_form'] = form

	return render(request, "account/login.html", context)


# Function Name:	logout_view
# Input:		    HTTP request
# Output:		    Logs out the user
# Logic:		    Use logout() function from django library
# Example Call:		Given by Django framework
def logout_view(request):
    logout(request)
    return redirect('homepage')


# Function Name:	register_view
# Input:		    HTTP request
# Output:		    Regsiters the new user
# Logic:		    Once the registration form is displayed, get the user email id, and password and login the user
#                   while storing the details in the database for future logins
def register_view(request):
    valuenext= request.POST.get('next')
    if request.POST:
        form = RegistrationForm(request.POST)
        if form.is_valid():
            account = form.save()
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('password')
            login(request, account)
            return redirect('homepage')
        
        else:
            return render(request, 'account/register.html', context={'registration_form': form})
    
    else:
        form = RegistrationForm()
        return render(request, 'account/register.html', context={'registration_form': form})


# Function Name:	about_us_view
# Input:		    HTTP request
# Output:		    Returns the template of the about us page
# Logic:		    Return and render the template
# Example Call:		Given by Django framework
def about_us_view(request):
    return render(request, 'account/about.html')


# Function Name:	contact_us_view
# Input:		    HTTP request
# Output:		    Returns the template of the contact us page
# Logic:		    Return and render the template
# Example Call:		Given by Django framework
def contact_us_view(request):
    return render(request, 'account/contact.html')


# Function Name:	account_view
# Input:		    HTTP request
# Output:		    User can change the username and email id if required
# Logic:		    Firstly, if the user is not logged in, ask him to login first
#                   Then get the current data from the database which will be editable
#                   and change the data. Finally display the updated data and also give a success message
# Example Call:		Given by Django framework
def account_view(request):
    
    context = {}
    if not request.user.is_authenticated:
        return redirect('login')
    if request.POST:
        form = AccountUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.initial = {
                'email': request.POST['email'],
                'username': request.POST['username'],
            }
            form.save()
            context['success_message'] = 'Changes saved!!!'
    
    else:
        form = AccountUpdateForm(
            initial = {
                'email': request.user.email,
                'username': request.user.username
            }
        )
    context['account_form'] = form
    
    return render(request, 'account/account.html', context)